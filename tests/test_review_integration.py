import math
import queue
import time
import numpy as np
import pytest

from autopress import diagnostics as dg
from autopress.acoustic import SpeechFeatureError, mfcc_features, find_outlier_sample
from autopress.evaluate import score_records
from autopress.neural import build_neural_template


def test_unlabelled_live_results_are_not_false_triggers_and_exports_are_explicit():
    log=dg.UtteranceLog(limit=2)
    live=dg.UtteranceRecord(uid='live',reason=dg.ACCEPTED,accepted='q')
    assert live.verdict()=='unlabelled'
    log.add(live)
    negative=dg.UtteranceRecord(uid='negative',reason=dg.ACCEPTED,accepted='q',expected='')
    log.add(negative)
    assert log.summary()['verdicts']=={'unlabelled':1,'false_trigger':1}
    assert log.export_features()==[]
    positive=dg.UtteranceRecord(uid='positive',reason=dg.REJECTED_FAR,top='q',expected='q',features=[[1.]*96],engine='neural')
    log.add(positive)
    assert len(log.records)==2
    assert positive.verdict()=='missed_right_top_far'
    assert len(log.export_features())==1
    log.clear();assert not log.records


def test_typed_errors_and_outlier_removal():
    with pytest.raises(SpeechFeatureError) as caught:mfcc_features(bytes(100),version=2)
    assert caught.value.code=='too_short'
    good=np.random.default_rng(4).normal(size=(30,24)).tolist()
    assert find_outlier_sample([good]*5) is None
    assert find_outlier_sample([good]*4+[good*2])==4


def test_offline_replay_supports_neural_and_rejects_wrong_feature_types():
    q=[1.]+[0.]*95;w=[0.,1.]+[0.]*94
    templates={'q':build_neural_template([[q]]*5),'w':build_neural_template([[w]]*5)}
    records=[{'expected':'q','features':[q]},{'expected':'','features':[q]}]
    result=score_records(templates,records)
    assert result['correct_accept']==1 and result['false_trigger']==1
    with pytest.raises(ValueError):score_records(templates,[{'expected':'q','features':[[1.]*24]}])


def test_replay_cli_is_read_only_and_rejects_changed_training(tmp_path,capsys):
    import json
    from autopress.evaluate import main
    from autopress.profiles import save_profiles,load_profiles
    from autopress.learning import training_id
    vector=[1.]+[0.]*95
    base={'q':build_neural_template([[vector]]*5)}
    profile_path=tmp_path/'profiles.json';validation_path=tmp_path/'validation.json'
    save_profiles(profile_path,{'LOL':{'commands':{'q':{'key':'Q'}},'neural_templates':base}})
    normalized=load_profiles(profile_path)['LOL']['neural_templates']
    rows=[{'expected':'q','features':[vector],'engine':'neural',
           'training_id':training_id(normalized),'learning_id':training_id({})}]
    validation_path.write_text(json.dumps({'records':rows}))
    before=profile_path.read_bytes()
    assert main([str(profile_path),'LOL',str(validation_path)])==0
    assert profile_path.read_bytes()==before
    assert 'selective' in capsys.readouterr().out
    rows[0]['training_id']='old-training'
    validation_path.write_text(json.dumps({'records':rows}))
    with pytest.raises(SystemExit) as error:main([str(profile_path),'LOL',str(validation_path)])
    assert error.value.code==2


def test_worker_wall_clock_includes_queue_delay_and_record_precedes_phrase(monkeypatch):
    from autopress import recognition
    from autopress.recognition import PatternRecognitionWorker, RecognitionWorker
    t=np.arange(4800)/16000
    voice=(np.sin(2*np.pi*180*t)*9000).astype('<i2').tobytes()
    features=mfcc_features(voice,version=2)
    templates={'q':{'version':2,'samples':[features]*5,'threshold':.5}}
    class Stream:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(recognition.sd,'RawInputStream',Stream)
    monkeypatch.setattr(RecognitionWorker,'_input_rate',lambda _:16000)
    class AcceptedMatcher:
        last_scores=[(.1,'q')]
        last_reason='kabul'
        thresholds={'q':.5}
        def __init__(self,*args):pass
        def match(self,features):return 'q',.1
    monkeypatch.setattr(recognition,'ShortCommandMatcher',AcceptedMatcher)
    worker=PatternRecognitionWorker(None,templates,.65,test_mode=True,capture_features=True)
    blocks=[voice[i:i+320] for i in range(0,len(voice),320)]+[bytes(320)]*40
    start=time.perf_counter()-2.
    source=iter((start+(i+1)*.01,block) for i,block in enumerate(blocks))
    def get(block=True,timeout=None):
        if not block:raise queue.Empty
        try:return next(source)
        except StopIteration:worker._stop.set();raise queue.Empty
    worker._audio.get=get
    events=[]
    worker.record.connect(lambda record:events.append(('record',record)))
    worker.phrase.connect(lambda *args:events.append(('phrase',args)))
    worker.run()
    assert worker._failure is None
    assert [kind for kind,_ in events]==['record','phrase']
    record=events[0][1]
    assert abs(record.speech_end_wall-(start+.3))<.02
    assert record.decision_ms>1000  # Timestamped backlog is retained in latency.
    assert 140<=record.endpoint_wait_ms<=180
    assert record.features is not None


def test_enrollment_does_not_accept_a_known_bad_sample_after_retry_limit(monkeypatch):
    from autopress import recognition
    from autopress.recognition import PatternRecognitionWorker,RecognitionWorker
    class Stream:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(recognition.sd,'RawInputStream',Stream)
    monkeypatch.setattr(RecognitionWorker,'_input_rate',lambda _:16000)
    monkeypatch.setattr(recognition,'find_outlier_sample',lambda samples:0)
    worker=PatternRecognitionWorker(None,{},.65,enrollment_phrase='q')
    t=np.arange(4800)/16000
    voice=(np.sin(2*np.pi*180*t)*9000).astype('<i2').tobytes()
    source=iter(([voice[i:i+320] for i in range(0,len(voice),320)]+[bytes(320)]*40)*8)
    def get(block=True,timeout=None):
        if not block:raise queue.Empty
        try:return next(source)
        except StopIteration:worker._stop.set();raise queue.Empty
    worker._audio.get=get
    trained=[];worker.trained.connect(lambda *args:trained.append(args))
    worker.run()
    assert not trained
    assert 'tutarsız' in worker._failure
