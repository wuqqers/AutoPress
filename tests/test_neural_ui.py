import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from autopress import app_tk
from autopress.neural import build_neural_template
from autopress.profiles import save_profiles, load_profiles


def test_neural_enrollment_ui_stages_promotes_and_keeps_legacy(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path/'profiles.json'
    old = {'version':2,'samples':[[[0.]*24]*6]*5,'threshold':.5}
    save_profiles(path, {'LOL':{'commands':{'q':{'key':'Q'},'w':{'key':'W'}},
                               'audio_templates':{'q':old,'w':old}}})
    monkeypatch.setattr(app_tk,'PROFILE_FILE',path)
    monkeypatch.setattr(app_tk.MainWindow,'refresh_devices',lambda self:None)
    window = app_tk.MainWindow()
    try:
        assert window._method() == 'neural'
        assert window._template_field() == 'neural_templates'
        for phrase,index in [('q',0),('w',1)]:
            window.select_command(phrase)
            window.open_enrollment()
            vector = [0.]*96; vector[index]=1.
            window._on_trained(phrase,build_neural_template([[vector]]*5))
            assert window.enr['done'].isEnabled(), window.status_label.text()
            saved = load_profiles(path)['LOL']
            assert saved['audio_templates']['q'] == old
            if phrase == 'q':
                assert 'q' in saved['pending_neural_templates']
                assert saved['neural_templates'] == {}
                assert '1/2' in window.pattern_status.text()
            dialog = window.enr['dialog']
            dialog.reject()
        saved = load_profiles(path)['LOL']
        assert set(saved['neural_templates']) == {'q','w'}
        assert saved['pending_neural_templates'] == {}
        assert '2/2' in window.pattern_status.text()
        window.method_menu.setCurrentText(app_tk.METHODS['pattern'])
        assert window._template_field() == 'audio_templates'
    finally:
        window.tray.hide()
        window.deleteLater()
        app.processEvents()
