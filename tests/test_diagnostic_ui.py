import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
import threading
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication
from autopress import app_tk,diagnostics as dg
from autopress.neural import build_neural_template
from autopress.learning import training_id
from autopress.profiles import save_profiles


def test_keyboard_outcomes_are_bound_to_the_matching_utterance_and_capture_freezes_learning(monkeypatch,tmp_path):
    app=QApplication.instance() or QApplication([])
    path=tmp_path/'profiles.json'
    vector=[1.]+[0.]*95
    templates={'q':build_neural_template([[vector]]*5)}
    save_profiles(path,{'LOL':{'commands':{'q':{'key':'Q','mode':'press'}},'neural_templates':templates}})
    monkeypatch.setattr(app_tk,'PROFILE_FILE',path)
    monkeypatch.setattr(app_tk.MainWindow,'refresh_devices',lambda self:None)
    win=app_tk.MainWindow()
    keys=[]
    win.keys=SimpleNamespace(held=set(),hold=lambda *args:keys.append('down'),release=lambda *args:keys.append('up'))
    monkeypatch.setattr(win,'_game_focused',lambda:False)
    def record(uid):
        return dg.UtteranceRecord(uid=uid,reason=dg.ACCEPTED,accepted='q',top='q',
            speech_end_wall=time.perf_counter()-.2,decision_wall=time.perf_counter()-.1)
    try:
        blocked=record('blocked');win._on_record(blocked);win._on_phrase('q',200)
        assert blocked.outcome==dg.OUT_BLOCKED_FOCUS and not keys
        win.is_test=True;win.expected_entry.setText('q')
        tested=record('test');win._on_record(tested);win._on_phrase('q',200)
        assert tested.outcome==dg.OUT_TEST_ONLY and not keys
        win.is_test=False;win.focus_check.setChecked(False)
        sent=record('sent');win._on_record(sent);win._on_phrase('q',200)
        assert sent.outcome==dg.OUT_KEY_SENT and sent.key_sent_wall is not None
        assert keys==['down','up'] and sent.e2e_ms>=200
        # A different phrase must not tag the old record.
        stale=record('stale');win._on_record(stale);win._on_phrase('unknown',200)
        assert stale.outcome==dg.OUT_PENDING
        win.capture_validation.setChecked(True)
        win.is_test=True
        win.worker=SimpleNamespace(_stop=threading.Event(),isRunning=lambda:True)
        for i in range(3):
            v=vector.copy();v[i+2]=.01*(i+1)
            import numpy as np
            v=(np.asarray(v)/np.linalg.norm(v)).tolist()
            win._on_learning_candidate('LOL',training_id(templates),{'id':str(i),'vector':v,
                'prediction':'q','suggestion':'q','auto_eligible':True,'reason':'kabul'})
        assert not win._profile().get('adaptive_learning',{})
        win._manual_learning(False)
        assert 'öğrenme durur' in win.learning_status.text()
        assert not win._profile().get('adaptive_learning',{})
    finally:
        win.worker=None;win.tray.hide();win.deleteLater();app.processEvents()
