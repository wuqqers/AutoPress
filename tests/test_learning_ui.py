import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import copy
import threading
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication
from autopress import app_tk
from autopress.learning import training_id
from autopress.neural import build_neural_template
from autopress.profiles import save_profiles, load_profiles


def vector(index,shift=0.):
    import numpy as np
    v=np.zeros(96);v[index]=1.;v[10]=shift
    return (v/np.linalg.norm(v)).tolist()


def test_feedback_auto_pause_undo_reset_and_io_failure(monkeypatch,tmp_path):
    app=QApplication.instance() or QApplication([])
    base={k:build_neural_template([[vector(i)]]*5) for k,i in [('q',0),('w',1)]}
    path=tmp_path/'profiles.json'
    save_profiles(path,{'LOL':{'commands':{'q':{'key':'Q'},'w':{'key':'W'}},
                             'neural_templates':base,'auto_learning':False}})
    monkeypatch.setattr(app_tk,'PROFILE_FILE',path)
    monkeypatch.setattr(app_tk.MainWindow,'refresh_devices',lambda self:None)
    win=app_tk.MainWindow();updates=[]
    win.worker=SimpleNamespace(_stop=threading.Event(),update_learning=updates.append,isRunning=lambda:False)
    win.worker_kind='neural'
    fingerprint=training_id(base)
    def offer(identity,index=0,shift=.005):
        win._on_learning_candidate('LOL',fingerprint,{'id':identity,'vector':vector(index,shift),
            'prediction':'q','suggestion':'q','reason':'kabul','auto_eligible':True})
    try:
        offer('wrong',1)
        win.feedback_label.setCurrentText('w');win._manual_learning(True)
        data=load_profiles(path)['LOL']['adaptive_learning']
        assert data['examples'][0]['label']=='w'
        assert data['blocked_labels']==['q']
        assert updates[-1]==data
        win._undo_learning()
        assert not load_profiles(path)['LOL']['adaptive_learning']
        offer('correct');win._manual_learning(False)
        assert load_profiles(path)['LOL']['adaptive_learning']['examples'][0]['source']=='manual'
        win._manual_learning(False)
        assert len(load_profiles(path)['LOL']['adaptive_learning']['examples'])==1
        win._reset_learning()
        assert load_profiles(path)['LOL']['neural_templates']==base
        assert not load_profiles(path)['LOL']['adaptive_learning']
        win.auto_learning_check.setChecked(True)
        for i,shift in enumerate((.005,.012,.02)): offer('auto'+str(i),0,shift)
        learned=load_profiles(path)['LOL']['adaptive_learning']
        assert len(learned['examples'])==1 and learned['examples'][0]['source']=='auto'
        win._reject_learning()
        rejected=load_profiles(path)['LOL']['adaptive_learning']
        assert not rejected['examples'] and rejected['blocked_labels']==['q']
        win.auto_learning_check.setChecked(False)
        before=copy.deepcopy(win.profiles)
        for i in range(4): offer('paused'+str(i),0,.03+i*.01)
        assert win.profiles==before
        before_reviews=len(win.review_events)
        win._on_learning_candidate('OTHER',fingerprint,{'id':'stale'})
        assert len(win.review_events)==before_reviews
        def fail(*args): raise OSError('disk full')
        monkeypatch.setattr(app_tk,'save_profiles',fail)
        win._manual_learning(False)
        assert win.profiles==before
        assert 'kaydedilemedi' in win.learning_status.text()
    finally:
        win.tray.hide();win.deleteLater();app.processEvents()
