import numpy as np
import pytest

from autopress.neural import (MODEL_ID, SpeechEncoder, NeuralMatcher,
                              build_neural_template, load_encoder)
from autopress.profiles import load_profiles, save_profiles, validate_profiles


def vector(index):
    x = np.zeros(96); x[index] = 1.
    return [x.tolist()]


def test_few_shot_head_classifies_new_vectors_and_rejects_unknown():
    rng = np.random.default_rng(5)
    templates = {}
    for key, index in [('q',0),('w',1),('e',2)]:
        samples = []
        for _ in range(5):
            x = np.asarray(vector(index))[0] + rng.normal(0,.005,96)
            x /= np.linalg.norm(x)
            samples.append([x.tolist()])
        templates[key] = build_neural_template(samples)
    matcher = NeuralMatcher(templates)
    for key,i in [('q',0),('w',1),('e',2)]:
        assert matcher.match(vector(i))[0] == key
    assert matcher.match(vector(80))[0] is None
    assert matcher.match([])[0] is None
    assert matcher.match([[float('nan')]*96])[0] is None


def test_single_class_rejects_unrelated_vector():
    matcher = NeuralMatcher({'q':build_neural_template([vector(0)]*5)})
    assert matcher.match(vector(0))[0] == 'q'
    assert matcher.match(vector(50))[0] is None


def test_model_identity_is_required():
    template = build_neural_template([vector(0)]*5)
    template['model_id'] = 'other-model'
    with pytest.raises(ValueError): NeuralMatcher({'q':template})


def test_neural_profiles_are_separate_and_survive_restart(tmp_path):
    old = {'version':2,'samples':[[[0.]*24]*6]*5,'threshold':.5}
    new = build_neural_template([vector(0)]*5)
    path = tmp_path/'profiles.json'
    data = {'LOL':{'commands':{'q':{'key':'Q'}},'audio_templates':{'q':old},
                   'pending_neural_templates':{'q':new}}}
    save_profiles(path,data)
    loaded = load_profiles(path)['LOL']
    assert loaded['audio_templates']['q'] == old
    assert loaded['pending_neural_templates']['q'] == new
    assert loaded['neural_templates'] == {}
    new['samples'][0] = [[0.]*95]
    with pytest.raises(ValueError): validate_profiles(data)


def test_real_pretrained_encoder_executes_and_checks_audio():
    pytest.importorskip('onnxruntime')
    encoder = load_encoder()
    assert encoder.encoder.get_providers() == ['CPUExecutionProvider']
    t = np.arange(4800)/16000
    pcm = (np.sin(2*np.pi*180*t)*np.sin(np.pi*np.arange(4800)/4800)*8000).astype('<i2').tobytes()
    first = np.asarray(encoder.encode(pcm))
    second = np.asarray(encoder.encode(pcm))
    assert first.shape == (1,96)
    assert np.isfinite(first).all()
    assert np.allclose(first,second)
    assert abs(np.linalg.norm(first)-1) < 1e-5
    for invalid in (b'', bytes(16000), np.full(2000,32767,dtype='<i2').tobytes()):
        with pytest.raises(ValueError): encoder.encode(invalid)


def test_missing_backbone_does_not_fall_back_to_mfcc(tmp_path):
    pytest.importorskip('onnxruntime')
    with pytest.raises(ValueError, match='eksik veya bozuk'): SpeechEncoder(tmp_path)


def test_real_neural_worker_enrolls_five_samples_then_recognizes(monkeypatch):
    pytest.importorskip('onnxruntime')
    import queue
    import autopress.recognition as recognition
    from autopress.recognition import PatternRecognitionWorker, RecognitionWorker

    class Stream:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(recognition.sd, 'RawInputStream', Stream)
    monkeypatch.setattr(RecognitionWorker, '_input_rate', lambda _:16000)
    def run(worker, repeats):
        t = np.arange(4800)/16000
        voice = (np.sin(2*np.pi*180*t)*np.sin(np.pi*np.arange(4800)/4800)*9000).astype('<i2').tobytes()
        chunks = iter(([voice[i:i+320] for i in range(0,len(voice),320)] + [bytes(320)]*50)*repeats)
        def read(block=True, timeout=None):
            if not block: raise queue.Empty
            try: return next(chunks)
            except StopIteration:
                worker._stop.set(); raise queue.Empty
        worker._audio.get = read
        worker.run()
        assert worker._failure is None

    trained = []
    enroll = PatternRecognitionWorker(None,{},.65,enrollment_phrase='q',enrollment_version=3)
    enroll.trained.connect(lambda phrase, template: trained.append((phrase,template)))
    run(enroll,5)
    assert len(trained) == 1
    assert trained[0][1]['version'] == 3
    assert len(trained[0][1]['samples']) == 5
    assert trained[0][1]['model_id'] == MODEL_ID
    assert len(trained[0][1]['phonetic_samples']) == 5
    detected = []
    worker = PatternRecognitionWorker(None,dict(trained),.65)
    from autopress import learning
    original_matcher = learning.AdaptiveMatcher
    updates = []
    class ObservedMatcher(original_matcher):
        def __init__(self, templates, mode='balanced', learning=None):
            updates.append(learning)
            super().__init__(templates,mode,learning)
    monkeypatch.setattr(learning,'AdaptiveMatcher',ObservedMatcher)
    def confirm(event):
        data = learning.add_example({},dict(trained),event,'q','manual')
        worker.update_learning(data)
    worker.learning_candidate.connect(confirm)
    worker.phrase.connect(lambda phrase, latency: detected.append(phrase))
    run(worker,2)
    assert detected == ['q','q']
    assert len(updates) >= 2 and updates[1]['examples'][0]['source'] == 'manual'
