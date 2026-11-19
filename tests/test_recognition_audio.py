from array import array
from pathlib import Path
import numpy as np
import pytest
import queue

import sounddevice as sd
import autopress.recognition as recognition

from autopress.recognition import RecognitionWorker, PatternRecognitionWorker, StableCommandGate


@pytest.mark.parametrize('rate', [8000, 16000, 22050, 44100, 48000, 96000])
def test_vector_resampling_preserves_pcm(rate):
    samples = np.random.default_rng(12).integers(-32768, 32768, 961, dtype=np.int16).tolist()
    source = array('h', samples).tobytes()
    if rate == 16000:
        expected = source
    else:
        output = []
        for index in range(max(1, round(len(samples) * 16000 / rate))):
            position = index * rate / 16000
            left = min(int(position), len(samples) - 1)
            right = min(left + 1, len(samples) - 1)
            fraction = position - left
            output.append(max(-32768, min(32767, round(samples[left] * (1 - fraction) + samples[right] * fraction))))
        expected = array('h', output).tobytes()
    assert RecognitionWorker._resample_pcm(source, rate) == expected


def test_resample_48khz_pcm_to_vosk_rate():
    source = array("h", [1234] * 480)
    converted = RecognitionWorker._resample_pcm(source.tobytes(), 48000)
    result = array("h")
    result.frombytes(converted)
    assert len(result) == 160
    assert set(result) == {1234}


def test_pattern_worker_uses_prepared_matcher_and_reports_timings(monkeypatch):
    worker = PatternRecognitionWorker(None, {}, .65)
    observed, timings, prepared = [], [], []
    worker.phrase.connect(lambda *args: observed.append(args))
    worker.timings.connect(lambda *args: timings.append(args))
    clock = iter(np.arange(0, 10, .1))
    monkeypatch.setattr(recognition.time, 'perf_counter', lambda: float(next(clock)))
    monkeypatch.setattr(RecognitionWorker, '_input_rate', lambda _: 16000)

    class Matcher:
        def __init__(self, templates, sensitivity):
            prepared.append((templates, sensitivity))

        def match(self, features):
            assert len(features) > 4
            worker.stop()
            return 'flash', .1

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']

        def __enter__(self):
            t = np.arange(1600) / 16000
            voiced = (np.sin(2 * np.pi * 180 * t) * 9000).astype('<i2').tobytes()
            for block in [voiced] * 5 + [bytes(3200)] * 8:
                self.callback(block, 1600, None, None)
            return self

        def __exit__(self, *args):
            pass

    # Avoid a hanging test if endpoint detection regresses.
    original_get = worker._audio.get
    def get_audio(block=True, timeout=None):
        try:
            return original_get(block=False)
        except queue.Empty:
            worker._stop.set()
            raise
    monkeypatch.setattr(worker._audio, 'get', get_audio)
    monkeypatch.setattr(recognition, 'PreparedTemplateMatcher', Matcher)
    monkeypatch.setattr(recognition.sd, 'RawInputStream', Stream)
    worker.run()
    assert worker._failure is None
    assert len(prepared) == 1
    assert observed[0][0] == 'flash'
    assert len(timings) == 1 and all(value >= 0 for value in timings[0])


def test_input_rate_falls_back_to_supported_default(monkeypatch):
    monkeypatch.setattr(sd, "query_devices", lambda device, kind: {"name": "test mic", "default_samplerate": 48000})
    accepted = []

    def check_input_settings(*, device, samplerate, channels, dtype):
        if samplerate != 48000:
            raise ValueError("unsupported rate")
        accepted.append(samplerate)

    monkeypatch.setattr(sd, "check_input_settings", check_input_settings)
    assert RecognitionWorker._input_rate(14) == 48000
    assert accepted == [48000]


def test_vosk_partial_requires_stability_and_only_fires_once_per_utterance():
    gate = StableCommandGate({"flash"})
    assert gate.partial("flash", 0.9, 0.6, 160) is None
    assert gate.partial("flash", 0.9, 0.6, 180) is None
    assert gate.partial("flash", 0.9, 0.6, 200) == "flash"
    assert gate.partial("flash", 0.99, 0.6, 240) is None
    assert not gate.final("flash", 0.99, 0.6)
    gate.reset()
    assert gate.final("flash", 0.9, 0.6)
    assert not gate.final("flash", 0.9, 0.6)


def test_command_calibration_can_accept_a_phrase_before_commands_exist():
    gate = StableCommandGate(set())
    assert gate.final("zırh", 0.9, 0.6, allow_any=True)
    assert not gate.final("zıpla", 0.9, 0.6, allow_any=True)


def test_vosk_model_is_reused_between_listening_sessions(monkeypatch, tmp_path):
    loads=[]
    monkeypatch.setattr(recognition,"Model",lambda path: loads.append(path) or object())
    first=recognition._load_cached_model(tmp_path)
    second=recognition._load_cached_model(tmp_path)
    assert first is second
    assert len(loads)==1


def test_personal_worker_discards_raw_audio_on_stop():
    worker=PatternRecognitionWorker(None,{},0.65,enrollment_phrase="flash")
    worker._audio.put(b"temporary audio bytes")
    worker.stop()
    assert worker._stop.is_set()
    assert worker._audio.empty()


def test_both_recognition_workers_publish_live_microphone_level():
    assert hasattr(RecognitionWorker, "level")
    assert hasattr(PatternRecognitionWorker, "level")


def test_confusable_vosk_commands_require_final_disambiguation():
    worker=RecognitionWorker(Path('.'),None,.65,
                             commands={"zırh":{},"zıpla":{}})
    assert {"zırh","zıpla"}.issubset(worker.confusable_commands)
    close={"alternatives":[{"text":"zıpla","confidence":0.62},{"text":"zırh","confidence":0.56}]}
    clear={"alternatives":[{"text":"zıpla","confidence":0.88},{"text":"zırh","confidence":0.3}]}
    assert worker._alternative_conflict(close,"zıpla")
    assert not worker._alternative_conflict(clear,"zıpla")
