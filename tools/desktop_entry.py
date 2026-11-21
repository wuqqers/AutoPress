"""Standalone Windows executable entry point."""
import sys


def smoke_test(report):
    import json
    import os
    import tempfile
    from pathlib import Path

    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from autopress import app_tk
    from autopress.acoustic import mfcc_features
    import numpy as np

    # Isolate startup validation from the user's saved profiles. No audio stream
    # is opened and no keyboard events are sent.
    with tempfile.TemporaryDirectory(prefix='autopress-check-') as directory:
        app_tk.APP = Path(directory)
        app_tk.PROFILE_FILE = app_tk.APP / 'profiles.json'
        application = QApplication([])
        application.setStyleSheet(app_tk.QSS)
        window = app_tk.MainWindow()
        assert window._method() == 'neural'
        assert window.scan_check.isChecked()
        assert window.silence_duration.value() == 150
        assert not window.grab().isNull()
        t = np.arange(8000) / 16000
        features = mfcc_features((np.sin(2 * np.pi * 180 * t) * 9000).astype('<i2').tobytes())
        assert len(features) > 4
        from autopress.neural import load_encoder, NeuralMatcher, build_neural_template
        encoder = load_encoder()
        vector = encoder.encode((np.sin(2 * np.pi * 180 * t) * 9000).astype('<i2').tobytes())
        assert np.asarray(vector).shape == (1, 96)
        matcher = NeuralMatcher({'check': build_neural_template([vector] * 5)})
        assert matcher.match(vector)[0] == 'check'
        from autopress.learning import AdaptiveMatcher, add_example
        templates = {'check': build_neural_template([vector] * 5)}
        learning = add_example({}, templates, {'id':'smoke','vector':vector[0]}, 'check', 'manual')
        assert AdaptiveMatcher(templates,learning=learning).match(vector)[0] == 'check'
        assert window.auto_learning_check.isChecked()
        assert window.log.summary()['n'] == 0
        assert not window.capture_validation.isChecked()
        QTimer.singleShot(100, application.quit)
        application.exec()
        window.tray.hide()
        Path(report).write_text(json.dumps({'ok': True, 'frozen': bool(getattr(sys, 'frozen', False)),
                                           'ui': 'PySide6', 'features': len(features),
                                           'neural': encoder.encoder.get_providers(),
                                           'embedding_dimension': len(vector[0]),
                                           'adaptive_learning': True,
                                           'diagnostics': True}), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--smoke-test':
        try:
            smoke_test(sys.argv[2])
        except Exception:
            import traceback
            from pathlib import Path
            Path(sys.argv[2]).write_text(traceback.format_exc(), encoding='utf-8')
            sys.exit(1)
    else:
        from autopress.app import main
        main()
