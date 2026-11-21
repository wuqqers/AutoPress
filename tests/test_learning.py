import copy
import numpy as np
import pytest

from autopress.learning import (AdaptiveMatcher, AutoLearningGate, add_example,
                               preserves_enrollment, training_id, validate_learning)
from autopress.neural import NeuralMatcher, build_neural_template
from autopress.profiles import load_profiles, save_profiles


def vector(index, shift=0.):
    v = np.zeros(96); v[index]=1.; v[10]=shift
    return (v/np.linalg.norm(v)).tolist()


def templates():
    return {k:build_neural_template([[vector(i)]]*5) for k,i in [('q',0),('w',1)]}


def event(identity, index=0, shift=.005, eligible=True):
    return {'id':identity,'vector':vector(index,shift),'prediction':'q',
            'auto_eligible':eligible,'reason':'kabul','suggestion':'q'}


def test_manual_adaptation_learns_new_pronunciation_without_changing_backbone_or_anchors():
    base = templates(); original=copy.deepcopy(base)
    probe = [vector(2,.012)]
    assert NeuralMatcher(base).match(probe)[0] is None
    data = {}
    for i,shift in enumerate((.01,.014)):
        data = add_example(data,base,event(str(i),2,shift),'q','manual')
    adaptive=AdaptiveMatcher(base,learning=data)
    assert adaptive.match(probe)[0] == 'q'
    assert not adaptive.auto_eligible(probe,'q')  # Student cannot certify itself.
    assert preserves_enrollment(base,data)
    assert base == original


def test_auto_gate_requires_three_distinct_strong_sayings_and_throttles():
    gate = AutoLearningGate()
    assert not gate.consider(event('weak',eligible=False),{},0)
    assert not gate.consider(event('1',shift=.005),{},0)
    assert not gate.consider(event('duplicate',shift=.005),{},1)
    assert not gate.consider(event('2',shift=.01),{},2)
    assert gate.consider(event('3',shift=.015),{},3)
    for i,s in enumerate((.02,.025,.03)):
        assert not gate.consider(event('throttle'+str(i),shift=s),{},4+i)


def test_auto_requires_frozen_teacher_agreement():
    model = AdaptiveMatcher(templates())
    assert model.auto_eligible([vector(0,.005)],'q')
    assert not model.auto_eligible([vector(0,.005)],'w')
    assert not model.auto_eligible([vector(2)],None)


def test_limits_manual_correction_persistence_and_new_training_reset(tmp_path):
    base=templates();data={}
    for i in range(20): data=add_example(data,base,event('a'+str(i)),'q','auto')
    assert len(data['examples'])==3
    for i in range(20): data=add_example(data,base,event('m'+str(i)),'q','manual')
    assert len(data['examples'])==11
    auto_id=next(e['id'] for e in data['examples'] if e['source']=='auto')
    data=add_example(data,base,event(auto_id,1),'w','manual')
    assert sum(e['id']==auto_id for e in data['examples'])==1
    assert next(e for e in data['examples'] if e['id']==auto_id)['label']=='w'
    path=tmp_path/'profiles.json'
    save_profiles(path,{'LOL':{'commands':{'q':{'key':'Q'},'w':{'key':'W'}},
                             'neural_templates':base,'adaptive_learning':data,'auto_learning':False}})
    loaded=load_profiles(path)['LOL']
    assert loaded['adaptive_learning']==data
    assert loaded['auto_learning'] is False
    new=copy.deepcopy(base); new['q']['samples'][0]=[vector(3)]
    assert validate_learning(data,new)=={}


def test_bad_feedback_cannot_break_originally_recognized_enrollment():
    base=templates();data={}
    for i in range(2): data=add_example(data,base,event(str(i),0,0.),'w','manual')
    assert not preserves_enrollment(base,data)


def test_invalid_adaptation_is_rejected():
    base=templates()
    for bad in ([float('nan')]*96,[0.]*96,[1.]*95):
        record=event('bad');record['vector']=bad
        with pytest.raises(ValueError): add_example({},base,record,'q','manual')
    with pytest.raises(ValueError): add_example({},base,event('bad'),'missing','auto')
