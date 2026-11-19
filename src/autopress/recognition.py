"""Local Vosk stream. Audio chunks stay in memory and are discarded promptly."""
from __future__ import annotations
import json, queue, threading, time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
import sounddevice as sd
import numpy as np
import vosk
from vosk import KaldiRecognizer, Model
from PySide6.QtCore import QThread, Signal
from .acoustic import SAMPLE_RATE, PreparedTemplateMatcher, build_template, mfcc_features
from .profiles import normalize

_MODEL_CACHE: dict[str, Model] = {}
_MODEL_LOCK = threading.Lock()

def _load_cached_model(path: Path) -> Model:
    key=str(path.resolve())
    with _MODEL_LOCK:
        model=_MODEL_CACHE.get(key)
        if model is None:
            model=Model(key); _MODEL_CACHE[key]=model
        return model

class StableCommandGate:
    """Only accept a profile command after stable partials; one fire per utterance."""
    def __init__(self, commands: set[str]):
        self.commands = commands
        self.reset()
    def reset(self):
        self.last_candidate = ""; self.streak = 0; self.fired = ""
    def partial(self, text: str, confidence: float, threshold: float, elapsed_ms: float) -> str | None:
        candidate = normalize(text)
        if candidate and candidate == self.last_candidate: self.streak += 1
        else: self.last_candidate, self.streak = candidate, (1 if candidate else 0)
        if (candidate in self.commands and self.streak >= 3 and elapsed_ms >= 140
                and confidence >= threshold and not self.fired):
            self.fired = candidate
            return candidate
        return None
    def final(self, text: str, confidence: float, threshold: float, allow_any: bool = False) -> bool:
        candidate = normalize(text)
        if (candidate in self.commands or (allow_any and bool(candidate))) and confidence >= threshold and not self.fired:
            self.fired = candidate
            return True
        return False

class RecognitionWorker(QThread):
    phrase = Signal(str, float)
    level = Signal(float)
    observed = Signal(str, str, float, float, str)  # method, text, score, end-to-end ms, stage
    status = Signal(str)
    failed = Signal(str)
    metrics = Signal(float, float)
    def __init__(self, model_dir: Path, device: int | None, sensitivity: float,
                 commands: dict | None = None, test_mode: bool = False):
        super().__init__()
        self.model_dir, self.device = model_dir, device
        self.sensitivity = sensitivity
        self.commands = {normalize(x) for x in (commands or {})}
        self.confusable_commands = set()
        for left in self.commands:
            for right in self.commands:
                if left != right and left[:1] == right[:1] and SequenceMatcher(None,left,right).ratio() >= 0.4:
                    self.confusable_commands.update((left,right))
        self.test_mode = test_mode
        self._stop = threading.Event()
        self._failure: str | None = None
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=12)
    def stop(self):
        self._stop.set()
        self._discard_audio()
        try: self._audio.put_nowait(b"")
        except queue.Full: pass
    def _discard_audio(self):
        while True:
            try: self._audio.get_nowait()
            except queue.Empty: return
    @staticmethod
    def _input_rate(device: int | None) -> int:
        try:
            info = sd.query_devices(device, "input")
        except Exception as exc:
            raise RuntimeError(f"Seçilen mikrofon bilgisi alınamadı: {exc}") from exc
        rates = [16000, int(info["default_samplerate"]), 48000, 44100, 32000, 22050]
        tried = set()
        for rate in rates:
            if rate in tried: continue
            tried.add(rate)
            try:
                sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
                return rate
            except Exception:
                continue
        raise RuntimeError(
            f"'{info['name']}' mikrofonu desteklenen bir ses biçiminde açılamadı. "
            "Mikrofon listesinden başka bir giriş aygıtı seçin veya Windows mikrofon izinlerini kontrol edin."
        )
    @staticmethod
    def _resample_pcm(chunk: bytes, source_rate: int) -> bytes:
        if source_rate == 16000: return chunk
        samples = np.frombuffer(chunk[:len(chunk) - (len(chunk) % 2)], dtype="<i2")
        if not samples.size: return b""
        output_count = max(1, round(len(samples) * 16000 / source_rate))
        position = np.arange(output_count, dtype=np.float64) * source_rate / 16000
        left = np.minimum(position.astype(np.intp), len(samples) - 1)
        right = np.minimum(left + 1, len(samples) - 1)
        fraction = position - left
        output = np.rint(samples[left] * (1 - fraction) + samples[right] * fraction)
        return np.clip(output, -32768, 32767).astype("<i2").tobytes()
    def _alternative_conflict(self,result: dict,chosen: str) -> bool:
        if chosen not in self.confusable_commands: return False
        alternatives=result.get("alternatives",[])
        scores={normalize(x.get("text","")):float(x.get("confidence",0)) for x in alternatives if x.get("text")}
        chosen_score=scores.get(chosen)
        if chosen_score is None: return False
        return any(other in self.commands and other != chosen and abs(chosen_score-score) < 0.18
                   for other,score in scores.items())
    def run(self):
        try:
            self.status.emit("Türkçe tanıma modeli yükleniyor")
            model = _load_cached_model(self.model_dir)
            grammar = json.dumps([*sorted(self.commands), "[unk]"], ensure_ascii=False) if self.commands else None
            rec = KaldiRecognizer(model, SAMPLE_RATE, grammar) if grammar else KaldiRecognizer(model, SAMPLE_RATE)
            rec.SetWords(True)
            rec.SetPartialWords(True)
            if hasattr(rec,"SetMaxAlternatives"): rec.SetMaxAlternatives(3)
            endpointer = getattr(vosk, "EndpointerMode", None)
            if endpointer is not None and hasattr(rec, "SetEndpointerMode"):
                rec.SetEndpointerMode(endpointer.SHORT)
            input_rate = self._input_rate(self.device)
            def callback(indata, frames, timing, status):
                if status: self.status.emit(str(status))
                chunk = bytes(indata)
                try: self._audio.put_nowait(chunk)
                except queue.Full:
                    try: self._audio.get_nowait()
                    except queue.Empty: pass
                    try: self._audio.put_nowait(chunk)
                    except queue.Full: pass
            with sd.RawInputStream(samplerate=input_rate, blocksize=0, latency="low", device=self.device,
                                   dtype="int16", channels=1, callback=callback):
                rate_note = "" if input_rate == 16000 else f" · giriş {input_rate} Hz, yerel dönüştürme etkin"
                self.status.emit(f"Dinleniyor · profil komutlarıyla sınırlı, kararlı ara sonuç açık{rate_note}")
                noise_floor = 80.0
                next_level = 0.0
                speech_started = None
                gate = StableCommandGate(self.commands)
                while not self._stop.is_set():
                    try: chunk = self._audio.get(timeout=.2)
                    except queue.Empty: continue
                    if not chunk: continue
                    chunk = self._resample_pcm(chunk, input_rate)
                    now = time.perf_counter()
                    pcm = np.frombuffer(chunk[:len(chunk) - len(chunk) % 2], dtype="<i2").astype(np.float64)
                    rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
                    if now >= next_level:
                        self.level.emit(min(1.0, rms / 32768.0))
                        next_level = now + 1 / 30
                    if rms > max(180.0, noise_floor * 2.5):
                        if speech_started is None: speech_started = now
                    else:
                        noise_floor = noise_floor * 0.98 + rms * 0.02
                    complete = rec.AcceptWaveform(chunk)
                    if complete:
                        result = json.loads(rec.Result())
                        alternatives=result.get("alternatives",[])
                        phrase = (result.get("text") or (alternatives[0].get("text","") if alternatives else "")).strip()
                        words = result.get("result", [])
                        confidence = ((sum(w.get("conf", 0) for w in words) / len(words)) if words else
                                      (float(alternatives[0].get("confidence",0)) if alternatives else 0.0))
                        norm = normalize(phrase)
                        threshold = 0.15 + self.sensitivity * 0.7
                        is_ambiguous=self._alternative_conflict(result,norm)
                        if norm in self.confusable_commands: threshold=max(threshold,0.80)
                        rejected=is_ambiguous or (norm in self.commands and confidence < threshold)
                        elapsed = (time.perf_counter() - speech_started) * 1000 if speech_started is not None else -1.0
                        if phrase and (self.test_mode or norm in self.commands):
                            self.observed.emit("vosk", phrase, confidence, elapsed, "reddedildi" if rejected else "son")
                        accepted = gate.final(phrase, confidence, threshold,
                                              allow_any=self.test_mode and not self.commands)
                        if phrase and not rejected and accepted:
                            self.phrase.emit(phrase, elapsed)
                        self.metrics.emit(elapsed, len(chunk) / 32000)
                        speech_started = None; gate.reset()
                    else:
                        partial = json.loads(rec.PartialResult())
                        phrase = partial.get("partial", "").strip()
                        norm = normalize(phrase)
                        words = partial.get("partial_result", [])
                        confidence = (sum(w.get("conf", 0) for w in words) / len(words)) if words else 0.0
                        threshold = 0.15 + self.sensitivity * 0.7
                        elapsed = (time.perf_counter() - speech_started) * 1000 if speech_started is not None else 0.0
                        if norm not in self.confusable_commands and gate.partial(phrase, confidence, threshold, elapsed):
                            self.phrase.emit(phrase, elapsed)
                            self.observed.emit("vosk", phrase, confidence, elapsed, "erken")
        except Exception as exc:
            self._failure = str(exc)
            self.failed.emit(self._failure)
        finally:
            self._discard_audio()
            if self._failure is None:
                self.status.emit("Durduruldu")


class PatternRecognitionWorker(QThread):
    """MFCC/DTW matcher. Audio exists only in a bounded RAM buffer during a session."""
    observed = Signal(str, str, float, float, str)
    phrase = Signal(str, float)
    level = Signal(float)
    progress = Signal(int, int)
    trained = Signal(str, object)
    status = Signal(str)
    failed = Signal(str)
    timings = Signal(float, float, float)  # feature extraction, matching, endpoint wait (ms)
    def __init__(self, device: int | None, templates: dict[str, dict], sensitivity: float,
                 *, enrollment_phrase: str | None = None, sample_count: int = 5, test_mode: bool = False,
                 silence_seconds: float = 0.15):
        super().__init__()
        self.device, self.templates, self.sensitivity = device, templates, sensitivity
        self.enrollment_phrase, self.sample_count, self.test_mode = enrollment_phrase, sample_count, test_mode
        self.silence_seconds = max(0.10, min(0.60, float(silence_seconds)))
        self._stop = threading.Event(); self._failure = None
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=24)
    def stop(self):
        self._stop.set()
        self._discard_audio()
    def _discard_audio(self):
        while True:
            try: self._audio.get_nowait()
            except queue.Empty: return
    def run(self):
        try:
            input_rate = RecognitionWorker._input_rate(self.device)
            matcher = PreparedTemplateMatcher(self.templates, self.sensitivity) if not self.enrollment_phrase else None
            samples = []
            pre_roll = deque(maxlen=3); pending = []; segment = []
            speaking = False; last_voice = 0.0; speech_started = None; noise_floor = 90.0
            next_level = 0.0
            segment_bytes = 0
            def callback(indata, frames, timing, status):
                if status: self.status.emit(str(status))
                try: self._audio.put_nowait(bytes(indata))
                except queue.Full:
                    try: self._audio.get_nowait()
                    except queue.Empty: pass
                    try: self._audio.put_nowait(bytes(indata))
                    except queue.Full: pass
            with sd.RawInputStream(samplerate=input_rate, blocksize=0, latency="low", device=self.device,
                                   dtype="int16", channels=1, callback=callback):
                self.status.emit("Kişisel ses örüntüsü dinleniyor · ses yalnız RAM'de")
                while not self._stop.is_set():
                    try: raw = self._audio.get(timeout=.2)
                    except queue.Empty: continue
                    if not raw: continue
                    chunk = RecognitionWorker._resample_pcm(raw, input_rate)
                    pcm = np.frombuffer(chunk, dtype="<i2")
                    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if pcm.size else 0.0
                    now = time.perf_counter(); threshold = max(180.0, noise_floor * 2.5)
                    if now >= next_level:
                        self.level.emit(min(1.0, rms / 32768.0))
                        next_level = now + 1 / 30
                    if not speaking:
                        if rms > threshold:
                            pending.append(chunk)
                            if len(pending) >= 2:
                                speaking = True; segment = list(pre_roll) + pending
                                segment_bytes = sum(len(x) for x in segment)
                                speech_started = now - segment_bytes / 32000
                                last_voice = now; pending = []
                        else:
                            pending.clear(); noise_floor = noise_floor * 0.98 + rms * 0.02
                            pre_roll.append(chunk)
                        continue
                    segment.append(chunk)
                    segment_bytes += len(chunk)
                    if rms > threshold: last_voice = now
                    duration = segment_bytes / 32000
                    if duration > 5.0:
                        self.status.emit("5 saniyeden uzun konuşma atlandı; kısa komut söyleyin.")
                        speaking = False; segment = []; pending = []; pre_roll.clear(); speech_started = None
                        continue
                    if now - last_voice < self.silence_seconds: continue
                    features_started = time.perf_counter()
                    try: features = mfcc_features(b"".join(segment))
                    except ValueError as exc:
                        self.status.emit(f"Ses örüntüsü alınamadı: {exc}")
                        features = None
                    if features is not None:
                        if self._stop.is_set(): return
                        if self.enrollment_phrase:
                            samples.append(features); self.progress.emit(len(samples), self.sample_count)
                            self.status.emit(f"Örnek {len(samples)}/{self.sample_count} alındı. Ses örneği silindi; yalnız özellikler RAM'de tutuluyor.")
                            if len(samples) >= self.sample_count:
                                self.trained.emit(self.enrollment_phrase, build_template(samples))
                                self.status.emit("Kişisel ses örüntüsü hazır; özellikler profile otomatik kaydediliyor.")
                                return
                        else:
                            matching_started = time.perf_counter()
                            match, distance = matcher.match(features)
                            matched_at = time.perf_counter()
                            latency = (matched_at - speech_started) * 1000 if speech_started is not None else -1.0
                            if match: self.phrase.emit(match, latency)
                            self.timings.emit((matching_started - features_started) * 1000,
                                              (matched_at - matching_started) * 1000,
                                              (now - last_voice) * 1000)
                            self.observed.emit("örüntü", match or "", distance, latency, "son")
                    speaking = False; segment = []; pending = []; pre_roll.clear(); speech_started = None
        except Exception as exc:
            self._failure = str(exc); self.failed.emit(self._failure)
        finally:
            self._discard_audio()
