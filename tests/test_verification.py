import queue
import numpy as np
import pytest

from autopress.acoustic import mfcc_features
from autopress.neural import build_neural_template
from autopress.profiles import save_profiles, load_profiles
from autopress.verification import verification_templates


def tone(hz):
    t=np.arange(4800)/16000
    return (np.sin(2*np.pi*hz*t)*np.sin(np.pi*np.arange(4800)/4800)*9000).astype('<i2').tobytes()


def test_paired_features_survive_save_and_take_priority(tmp_path):
    pair=[mfcc_features(tone(180),version=2)]*5
    vector=[1.]+[0.]*95
    template=build_neural_template([[vector]]*5,pair)
    path=tmp_path/'profiles.json'
    save_profiles(path,{'LOL':{'commands':{'q':{'key':'Q'}},'neural_templates':{'q':template}}})
    saved=load_profiles(path)['LOL']['neural_templates']
    assert saved['q']['phonetic_samples']==pair
    assert verification_templates(saved,{})['q']['samples']==pair
    assert not verification_templates({**saved,'w':build_neural_template([[vector]]*5)}, {})


@pytest.mark.parametrize('neural_prediction,expected',[('w',[]),('q',['q'])])
def test_neural_candidate_requires_independent_audio_agreement(monkeypatch,neural_prediction,expected):
    pytest.importorskip('onnxruntime')
    from autopress import recognition,learning
    from autopress.recognition import PatternRecognitionWorker,RecognitionWorker
    vector=[1.]+[0.]*95
    neural={k:build_neural_template([[vector]]*5) for k in ('q','w')}
    acoustic={k:{'version':2,'samples':[mfcc_features(tone(hz),version=2)]*5,'threshold':.5}
              for k,hz in [('q',180),('w',760)]}
    class ForcedMatcher:
        last_reason='kabul'
        last_scores=[(.1,neural_prediction)]
        def __init__(self,*args): pass
        def match(self,features):return neural_prediction,.1
        def auto_eligible(self,features,prediction):return prediction is not None
    class Stream:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(learning,'AdaptiveMatcher',ForcedMatcher)
    monkeypatch.setattr(recognition.sd,'RawInputStream',Stream)
    monkeypatch.setattr(RecognitionWorker,'_input_rate',lambda _:16000)
    worker=PatternRecognitionWorker(None,neural,.65,phonetic_templates=acoustic)
    pcm=tone(180)
    chunks=iter([pcm[i:i+320] for i in range(0,len(pcm),320)]+[bytes(320)]*50)
    def read(block=True,timeout=None):
        if not block:raise queue.Empty
        try:return next(chunks)
        except StopIteration:worker._stop.set();raise queue.Empty
    worker._audio.get=read
    detected=[];candidates=[]
    worker.phrase.connect(lambda phrase,latency:detected.append(phrase))
    worker.learning_candidate.connect(candidates.append)
    worker.run()
    assert worker._failure is None
    assert detected==expected
    assert len(candidates)==1
    if not expected:
        assert candidates[0]['reason']=='ses kontrolleri uyuşmadı'
        assert not candidates[0]['auto_eligible']
        assert candidates[0]['suggestion']=='q'
