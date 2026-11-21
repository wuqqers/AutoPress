# Build with: python -m PyInstaller --noconfirm AutoPress.spec
from PyInstaller.utils.hooks import collect_dynamic_libs
from pathlib import Path
import PySide6

a = Analysis(
    ['tools/desktop_entry.py'],
    pathex=['src'],
    binaries=collect_dynamic_libs('vosk'),
    datas=[('src/autopress/resources/neural/*', 'resources/neural'), ('THIRD_PARTY_NOTICES.md', '.')],
    hiddenimports=['_cffi_backend'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['pytest', 'tkinter'],
    noarchive=False,
)
# Python distributions may ship an older CRT than Qt requires. The executable
# loads the root copy first, so use Qt's redistributable there as well.
qt_directory = Path(PySide6.__file__).parent
runtime_names = {'vcruntime140.dll', 'vcruntime140_1.dll'}
a.binaries = [
    (target, str(qt_directory / Path(source).name), kind)
    if Path(target).name.lower() in runtime_names and (qt_directory / Path(source).name).exists()
    else (target, source, kind)
    for target, source, kind in a.binaries
    # Qt uses Windows' unversioned ICU API. A third-party ICU found on PATH
    # exports version-suffixed symbols and must not shadow the system DLL.
    if Path(target).name.lower() not in {'ucrtbase.dll', 'icuuc.dll', 'icudt78.dll'}
]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='AutoPress',
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
