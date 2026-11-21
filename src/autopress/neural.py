"""Pretrained speech CNN plus a locally fitted few-shot classification head.

Google speech_embedding/1, converted by openWakeWord v0.5.1. No remote
inference, audio persistence, random backbone weights or MFCC substitution.
"""
from pathlib import Path
import hashlib
import sys
import threading

import numpy as np
from .acoustic import SpeechFeatureError

MODEL_ID = 'google-speech-embedding-oww-v0.5.1-96-mean-v1'
HASHES = {
    'melspectrogram.onnx': 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f',
    'embedding_model.onnx': '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f',
}
_cache = None
_lock = threading.Lock()


def model_directory():
    root = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    return root / 'resources' / 'neural'


class SpeechEncoder:
    def __init__(self, directory=None):
        import onnxruntime as ort
        directory = Path(directory) if directory else model_directory()
        for name, digest in HASHES.items():
            path = directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Sinir ağı modeli eksik veya bozuk: {name}. Uygulama paketini yeniden alın.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.mel = ort.InferenceSession(str(directory / 'melspectrogram.onnx'), options,
                                       providers=['CPUExecutionProvider'])
        self.encoder = ort.InferenceSession(str(directory / 'embedding_model.onnx'), options,
                                           providers=['CPUExecutionProvider'])

    def encode(self, pcm):
        audio = np.frombuffer(pcm[:len(pcm)//2*2], dtype='<i2').astype(np.float32)
        if len(audio) < 960:
            raise SpeechFeatureError('Konuşma çok kısa; harfin adını doğal biçimde söyleyin.', 'too_short')
        if len(audio) > 80000:
            raise SpeechFeatureError('Tek komut en fazla 5 saniye olabilir.', 'too_long')
        if np.mean(np.abs(audio) >= 32600) > .01:
            raise SpeechFeatureError('Mikrofon sesi kırpılıyor; giriş sesini azaltın.', 'clipped')
        # Trim with energy windows; retain 20 ms around low-energy consonants.
        windows = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160]
        energy = np.sqrt(np.mean(windows * windows, axis=1))
        active = np.flatnonzero(energy > max(98., float(energy.max()) * .04))
        if len(active) < 4:
            raise SpeechFeatureError('Konuşma seviyesi çok düşük; yeniden söyleyin.', 'too_quiet')
        audio = audio[max(0, int(active[0])*160-320):min(len(audio),int(active[-1])*160+720)]
        # Unit gain reduces microphone-level variation without altering timing.
        audio *= min(4., 5000. / max(float(np.sqrt(np.mean(audio*audio))), 1.))
        audio = np.clip(audio, -32768, 32767)
        if len(audio) < 12800:
            missing = 12800 - len(audio)
            audio = np.pad(audio, (missing//2, missing-missing//2))
        spec = self.mel.run(None, {'input': audio[None].astype(np.float32)})[0].reshape(-1, 32)
        spec = spec / 10. + 2.
        starts = list(range(0, len(spec)-75, 38))
        if starts[-1] != len(spec)-76: starts.append(len(spec)-76)
        batch = np.stack([spec[i:i+76] for i in starts])[..., None].astype(np.float32)
        embeddings = self.encoder.run(None, {'input_1': batch})[0].reshape(-1, 96)
        vector = embeddings.mean(axis=0)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
            raise ValueError('Sinir ağı geçerli ses özelliği üretemedi.')
        vector /= np.linalg.norm(vector)
        return [vector.tolist()]


def load_encoder():
    global _cache
    with _lock:
        if _cache is None: _cache = SpeechEncoder()
        return _cache


def build_neural_template(samples, phonetic_samples=None):
    if len(samples) != 5:
        raise ValueError('Sinir ağı için beş geçerli örnek gerekli.')
    template = {'version': 3, 'model_id': MODEL_ID, 'samples': samples, 'threshold': .5}
    if phonetic_samples is not None: template['phonetic_samples'] = phonetic_samples
    return template


class NeuralMatcher:
    """Frozen speech CNN + ridge-trained dense head, with open-set rejection."""
    def __init__(self, templates, mode='balanced'):
        if not templates or any(t.get('model_id') != MODEL_ID or t.get('version') != 3 for t in templates.values()):
            raise ValueError('Sinir ağı örnekleri uyumsuz; komutları bu motorla yeniden öğretin.')
        self.mode = mode
        self.keys = list(templates)
        self.raw = np.array([s[0] for t in templates.values() for s in t['samples']], np.float64)
        self.labels = np.array([i for i,t in enumerate(templates.values()) for _ in t['samples']])
        if self.raw.ndim != 2 or self.raw.shape[1] != 96 or not np.isfinite(self.raw).all():
            raise ValueError('Sinir ağı örnek boyutu geçersiz.')
        if any(len(t['samples']) < 3 for t in templates.values()):
            raise ValueError('Her komuta en az üç sinir ağı örneği gerekli.')
        self.mean = self.raw.mean(axis=0) if len(self.keys) > 1 else np.zeros(96)
        self.scale = np.maximum(self.raw.std(axis=0), .03)
        self.x = self.normalize(self.raw)
        targets = np.eye(len(self.keys))[self.labels]
        # Dual ridge solve keeps training small with only five shots per class.
        self.weights = self.x.T @ np.linalg.solve(self.x @ self.x.T + .3*np.eye(len(self.x)), targets)
        self.thresholds = {}
        self.warnings = []
        self.last_scores = []
        self.last_reason = ''
        for i,key in enumerate(self.keys):
            own = self.x[self.labels == i]
            distances = 1. - own @ own.T
            np.fill_diagonal(distances, np.inf)
            holdout = np.sort(distances, axis=1)[:, :2].mean(axis=1)
            self.thresholds[key] = min(1.1, max(.15, float(np.quantile(holdout,.9))*1.4+.05))
            others = self.x[self.labels != i]
            if len(others) and float(np.min(1.-own @ others.T)) < float(np.median(holdout)):
                self.warnings.append(key)

    def normalize(self, vectors):
        x = (vectors-self.mean)/self.scale
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def match(self, features):
        vector = np.asarray(features, np.float64)
        if vector.shape != (1,96) or not np.isfinite(vector).all():
            self.last_reason = 'geçersiz'; return None, float('inf')
        x = self.normalize(vector)[0]
        if np.linalg.norm(x) < .5:
            self.last_reason = 'geçersiz'; return None,float('inf')
        logits = x @ self.weights
        winner = int(np.argmax(logits))
        distances = 1. - self.x @ x
        scores = sorted((float(np.mean(np.sort(distances[self.labels==i])[:2])), key)
                        for i,key in enumerate(self.keys))
        self.last_scores = scores[:2]
        distance, key = scores[0]
        factor, margin = {'selective': (.85,.12), 'balanced': (1.,.07), 'aggressive': (1.15,.03)}[self.mode]
        if distance > self.thresholds[key]*factor:
            self.last_reason = 'uzak'; return None,distance
        if self.keys[winner] != key or (len(scores)>1 and scores[1][0]-distance < margin):
            self.last_reason = 'kararsız'; return None,distance
        self.last_reason = 'kabul'
        return key,distance
