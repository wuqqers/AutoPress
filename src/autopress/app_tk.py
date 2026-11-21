"""AutoPress masaüstü arayüzü (PySide6)."""
from __future__ import annotations

import ctypes
import copy
import math
import os
import random
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QStyle, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

from .keys import KeyController
from . import diagnostics as dg
from .profiles import load_profiles, normalize, save_profiles
from .recognition import PatternRecognitionWorker, RecognitionWorker

APP = Path(os.getenv("LOCALAPPDATA", Path.home())) / "AutoPress"
_std = Path.home() / "AppData" / "Local" / "AutoPress"
if not (APP / "vosk-model-small-tr-0.3").is_dir() and (_std / "vosk-model-small-tr-0.3").is_dir():
    APP = _std
MODEL = APP / "vosk-model-small-tr-0.3"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip"
PROFILE_FILE = APP / "profiles.json"

AMBER, DIM, LINE = "#ffb020", "#8d8374", "#2b2621"
METHODS = {"neural": "Kişisel sinir ağı (yeni)", "pattern": "Kişisel örüntü (MFCC/DTW)", "vosk": "Vosk komut tanıma"}

def window_title_matches(expected: str, actual: str) -> bool:
    """Boş filtre, ön plandaki tüm pencerelere izin verir."""
    expected = expected.strip().casefold()
    return not expected or expected in actual.casefold()

QSS = f"""
QWidget{{color:#efe7da;font-family:'Bahnschrift','Segoe UI';font-size:13px}}
QMainWindow,#root{{background:transparent}}
QDialog{{background:#12100e}}
QLabel{{background:transparent}}
QLabel#h1{{font-size:24px;font-weight:600;letter-spacing:2px}}
QLabel#h2{{font-size:18px;font-weight:600;letter-spacing:1px}}
QLabel#dim{{color:{DIM}}}
QLabel#mono{{font-family:'Cascadia Mono','Consolas';color:{DIM};font-size:11px;letter-spacing:1px}}
QLabel#rowt{{font-size:17px;font-weight:600}}
QLabel#hero{{font-size:38px;font-weight:600;letter-spacing:1px}}
QFrame#panel{{background:rgba(24,21,18,215);border:1px solid {LINE};border-radius:16px}}
QFrame#row{{background:rgba(29,26,23,200);border:1px solid {LINE};border-radius:12px}}
QFrame#row:hover{{border-color:#5a4a2a}}
QFrame#row[sel="true"]{{border:1px solid {AMBER};background:rgba(58,42,12,150)}}
QPushButton{{background:#1d1a17;border:1px solid {LINE};border-radius:9px;padding:9px 16px;font-weight:600;letter-spacing:1px}}
QPushButton:hover,QPushButton:focus{{border-color:{AMBER}}}
QPushButton:disabled{{color:#5d554a;border-color:#211d19}}
QPushButton#primary{{background:{AMBER};color:#1a1204;border:1px solid {AMBER}}}
QPushButton#primary:hover{{background:#ffc652}}
QPushButton#primary:disabled{{background:#3a2e14;color:#7a6a45;border-color:#3a2e14}}
QPushButton#danger{{color:#ff7b7f}}
QPushButton#danger:hover{{border-color:#e5484d}}
QPushButton#nav{{background:transparent;border:none;border-bottom:2px solid transparent;border-radius:0;
  padding:10px 18px;color:{DIM};font-family:'Cascadia Mono','Consolas';letter-spacing:2px}}
QPushButton#nav:checked{{color:{AMBER};border-bottom:2px solid {AMBER}}}
QPushButton#seg{{border-radius:0;padding:9px 14px}}
QPushButton#seg:checked{{background:#3a2a0c;color:{AMBER};border-color:{AMBER}}}
QPushButton#cap{{font-family:'Cascadia Mono','Consolas';min-width:150px}}
QLineEdit,QComboBox{{background:#100e0c;border:1px solid {LINE};border-radius:9px;padding:8px 10px;
  selection-background-color:{AMBER};selection-color:#1a1204}}
QLineEdit:focus,QComboBox:focus{{border-color:{AMBER}}}
QComboBox QAbstractItemView{{background:#151311;border:1px solid {LINE};selection-background-color:#3a2a0c;outline:0}}
QScrollArea{{border:none;background:transparent}}
QScrollArea>QWidget>QWidget{{background:transparent}}
QScrollBar:vertical{{background:transparent;width:8px;margin:2px}}
QScrollBar::handle:vertical{{background:#3a342c;border-radius:4px;min-height:30px}}
QScrollBar::add-line,QScrollBar::sub-line{{height:0}}
QSlider::groove:horizontal{{height:4px;background:{LINE};border-radius:2px}}
QSlider::sub-page:horizontal{{background:{AMBER};border-radius:2px}}
QSlider::handle:horizontal{{background:{AMBER};width:16px;margin:-6px 0;border-radius:8px}}
QCheckBox{{spacing:10px}}
QCheckBox::indicator{{width:18px;height:18px;border:1px solid #4a4238;border-radius:5px;background:#100e0c}}
QCheckBox::indicator:checked{{background:{AMBER};border-color:{AMBER}}}
QToolTip{{background:#1d1a17;color:#efe7da;border:1px solid {LINE}}}
"""


class Backdrop(QWidget):
    """Mat zemin + amber ışık sızıntısı + film greni."""
    _grain: QPixmap | None = None

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0c0b0a"))
        g = QRadialGradient(QPointF(self.width() * .12, -40), self.width() * .75)
        g.setColorAt(0, QColor(255, 176, 32, 48)); g.setColorAt(1, QColor(255, 176, 32, 0))
        p.fillRect(self.rect(), g)
        if Backdrop._grain is None:
            img = QImage(160, 160, QImage.Format_ARGB32)
            for x in range(160):
                for y in range(160):
                    v = random.randint(0, 255); img.setPixelColor(x, y, QColor(v, v, v, 13))
            Backdrop._grain = QPixmap.fromImage(img)
        p.drawTiledPixmap(self.rect(), Backdrop._grain)


class Scope(QWidget):
    """Mikrofon ses seviyesini yumuşatarak gösteren canlı osiloskop."""
    N = 64

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(130)
        self.mode, self.pulse, self.t, self.bars, self.input_level = "off", 0.0, 0.0, [0.0] * self.N, 0.0
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(33)

    def set_mode(self, mode):
        self.mode = mode
        if mode == "off": self.input_level = 0.0
    @Slot(float)
    def amp(self, level): self.input_level = max(0.0, min(1.0, float(level)))
    def trigger(self): self.pulse = 1.0

    def _tick(self):
        self.t += .06; self.pulse *= .93
        level = min(1.0, self.input_level * 6.0) if self.mode != "off" else 0.0
        for i in range(self.N):
            wave = .18 + .82 * abs(math.sin(self.t * .9 + i * .19))
            spike = self.pulse * .75 * math.exp(-((i - self.N / 2) / (self.N * .18)) ** 2)
            target = level * wave + spike
            self.bars[i] += (target - self.bars[i]) * .45
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
        w, h = self.width(), self.height(); bw = w / self.N
        c = QColor("#3a342c") if self.mode == "off" else QColor(AMBER)
        for i, a in enumerate(self.bars):
            bh = max(3.0, a * h * .92)
            c.setAlphaF(.35 + .65 * min(1.0, a * 2.2) if self.mode != "off" else 1.0)
            p.setBrush(c); p.drawRoundedRect(QRectF(i * bw + bw * .22, (h - bh) / 2, bw * .56, bh), 2, 2)


class Keycap(QWidget):
    """Fiziksel klavye tuşu; komut tetiklenince ışıkla 'basılır'."""

    def __init__(self, text="", w=64, h=54):
        super().__init__()
        self.text, self.lit = text, 0.0
        self.setFixedSize(w, h); self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.timer = QTimer(self); self.timer.timeout.connect(self._decay)

    def set_text(self, text): self.text = text; self.update()
    def flash(self): self.lit = 1.0; self.timer.start(30)

    def _decay(self):
        self.lit *= .9
        if self.lit < .02: self.lit = 0.0; self.timer.stop()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h, dip = self.width(), self.height(), 4 * self.lit
        p.setBrush(QColor("#0a0908")); p.setPen(Qt.NoPen); p.drawRoundedRect(QRectF(0, 5, w, h - 5), 9, 9)
        face = QColor("#2a2520"); lit = QColor(AMBER)
        mix = lambda a, b: int(a + (b - a) * self.lit)
        p.setBrush(QColor(mix(face.red(), lit.red()), mix(face.green(), lit.green()), mix(face.blue(), lit.blue())))
        p.setPen(QColor("#4a4238") if self.lit < .05 else QColor("#ffd27a"))
        p.drawRoundedRect(QRectF(1, 1 + dip, w - 2, h - 9), 9, 9)
        f = QFont("Cascadia Mono", 11 if len(self.text) <= 5 else 8); f.setBold(True); p.setFont(f)
        p.setPen(QColor("#1a1204") if self.lit > .5 else QColor("#efe7da"))
        p.drawText(QRectF(0, dip, w, h - 8), Qt.AlignCenter, self.text or "—")


SPECIAL = {Qt.Key_Space: "SPACE", Qt.Key_Control: "CTRL", Qt.Key_Shift: "SHIFT", Qt.Key_Alt: "ALT",
           Qt.Key_Tab: "TAB", Qt.Key_Return: "ENTER", Qt.Key_Enter: "ENTER", Qt.Key_Escape: "ESC",
           Qt.Key_Up: "UP", Qt.Key_Down: "DOWN", Qt.Key_Left: "LEFT", Qt.Key_Right: "RIGHT"}


class KeyCapture(QPushButton):
    """Tıkla, bir tuşa bas: ad + Windows sanal tuş kodu yakalanır. Tuş oyuna gönderilmez."""
    captured = Signal(str, int)

    def __init__(self):
        super().__init__("TUŞU YAKALA")
        self.setObjectName("cap"); self.armed = False; self.key = ""; self.vk = None
        self.clicked.connect(self._arm)

    def set_key(self, key, vk):
        self.key, self.vk, self.armed = key, vk, False
        self.setText(f"{key} · {vk}" if vk is not None else (key or "TUŞU YAKALA"))

    def _arm(self): self.armed = True; self.setText("● BİR TUŞA BASIN"); self.setFocus()
    def focusNextPrevChild(self, nxt): return False if self.armed else super().focusNextPrevChild(nxt)
    def focusOutEvent(self, e):
        if self.armed: self.set_key(self.key, self.vk)
        super().focusOutEvent(e)

    def keyPressEvent(self, e):
        if not self.armed: return super().keyPressEvent(e)
        if e.isAutoRepeat(): return
        k, code = e.key(), int(e.nativeVirtualKey())
        if k in SPECIAL: name = SPECIAL[k]
        elif Qt.Key_F1 <= k <= Qt.Key_F12: name = f"F{k - Qt.Key_F1 + 1}"
        elif Qt.Key_0 <= k <= Qt.Key_9 or Qt.Key_A <= k <= Qt.Key_Z: name = chr(k)
        else: name = f"VK_{code}"
        self.set_key(name, code or None); self.captured.emit(name, code)


class CommandRow(QFrame):
    clicked = Signal(str)

    def __init__(self, phrase, key, mode, ready, selected):
        super().__init__()
        self.phrase = phrase; self.setObjectName("row"); self.setProperty("sel", selected)
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self); lay.setContentsMargins(12, 10, 14, 10); lay.setSpacing(14)
        lay.addWidget(Keycap(key, 62, 50))
        col = QVBoxLayout(); col.setSpacing(2)
        t = QLabel(phrase); t.setObjectName("rowt")
        m = QLabel(("KISA BASIŞ" if mode == "press" else "SÖYLEYİNCE BIRAK") + "   ·   " + (ready if isinstance(ready, str) else "ÖRÜNTÜ HAZIR" if ready else "ÖRÜNTÜ YOK"))
        m.setObjectName("mono"); col.addWidget(t); col.addWidget(m); lay.addLayout(col, 1)
        dot = QLabel("●"); dot.setStyleSheet(f"color:{AMBER if ready else '#3a342c'};font-size:16px"); lay.addWidget(dot)

    def mousePressEvent(self, e): self.clicked.emit(self.phrase)


def label(text, name=None, wrap=False):
    l = QLabel(text)
    if name: l.setObjectName(name)
    l.setWordWrap(wrap)
    return l


def panel():
    f = QFrame(); f.setObjectName("panel")
    return f


def scroll(widget):
    s = QScrollArea(); s.setWidgetResizable(True); s.setWidget(widget); s.setFrameShape(QFrame.NoFrame)
    return s


class MainWindow(QMainWindow):
    download_done = Signal(bool, str)
    STATES = {"off": ("DİNLEME KAPALI", "off"), "live": ("DİNLİYOR", "listen"),
              "test": ("GÜVENLİ DENEME · TUŞ GÖNDERİLMEZ", "test"), "enroll": ("ÖRNEK ALINIYOR", "enroll"),
              "wizard": ("YENİ KOMUT · TUŞ GÖNDERİLMEZ", "test")}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoPress | Sesle oyun kontrolü"); self.resize(1180, 780); self.setMinimumSize(980, 660)
        self.keys = KeyController()
        self.worker = None; self.worker_kind = None; self.enr = None
        self.log = dg.UtteranceLog()
        self._pending_record = None
        self._test_expected = None
        self.review_events = {}
        self.learning_undo = []
        from .learning import AutoLearningGate
        self.auto_gate = AutoLearningGate()
        self.validation_counts = {}
        self.last_key_sent_at = None
        self.metrics: list[float] = []; self.last_command = ""; self.last_time = 0.0; self.is_test = False
        self.selected = ""; self.device_index = None; self.device_labels = {"Varsayılan mikrofon": None}
        self.profiles = load_profiles(PROFILE_FILE)
        if not PROFILE_FILE.exists(): save_profiles(PROFILE_FILE, self.profiles)
        self.profile_name = next(iter(self.profiles))
        self._build()
        self._update_learning_totals()
        self.method_menu.currentTextChanged.connect(self._method_changed)
        self.download_done.connect(self._on_download)
        self.target_entry.setText(self._profile().get("target_window", ""))
        self.window_pick_deadline = 0.0
        self.window_pick_timer = QTimer(self); self.window_pick_timer.setInterval(150)
        self.window_pick_timer.timeout.connect(self._poll_window_pick)
        self._setup_tray()
        self.refresh_devices(); self.refresh_commands(); self.update_pattern_status()
        for ms, fn in ((400, self._guard_focus), (1000, self._update_resources)):
            t = QTimer(self); t.timeout.connect(fn); t.start(ms)

    # ---------- UI ----------
    def _build(self):
        root = Backdrop(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(24, 20, 24, 20); outer.setSpacing(16)

        top = QHBoxLayout(); top.setSpacing(12)
        brand = QVBoxLayout(); brand.setSpacing(0)
        brand.addWidget(label("AUTOPRESS", "h1")); brand.addWidget(label("SESLE OYUN KONTROLÜ · ERİŞİLEBİLİRLİK ARACI", "mono"))
        top.addLayout(brand); top.addStretch(1)
        self.profile_menu = QComboBox(); self.profile_menu.addItems(list(self.profiles)); self.profile_menu.setCurrentText(self.profile_name)
        self.profile_menu.currentTextChanged.connect(self._change_profile); self.profile_menu.setMinimumWidth(160)
        add = QPushButton("+ PROFİL"); add.clicked.connect(self.add_profile)
        self.device_menu = QComboBox(); self.device_menu.setMinimumWidth(220)
        self.device_menu.currentTextChanged.connect(lambda t: setattr(self, "device_index", self.device_labels.get(t)))
        self.method_menu = QComboBox(); self.method_menu.addItems(list(METHODS.values()))
        self.method_menu.setCurrentText(METHODS["neural"])
        for cap, w in (("OYUN PROFİLİ", self.profile_menu), ("", add), ("MİKROFON", self.device_menu), ("YÖNTEM", self.method_menu)):
            col = QVBoxLayout(); col.setSpacing(3)
            if cap: col.addWidget(label(cap, "mono"))
            else: col.addSpacing(17)
            col.addWidget(w); top.addLayout(col)
        outer.addLayout(top)

        body = QHBoxLayout(); body.setSpacing(16); outer.addLayout(body, 1)
        body.addWidget(self._build_hero())
        right = QVBoxLayout(); right.setSpacing(10); body.addLayout(right, 1)
        nav = QHBoxLayout(); nav.setSpacing(0); self.nav_group = QButtonGroup(self); self.stack = QStackedWidget()
        pages = (("KOMUTLAR", self._page_commands()), ("DENEME", self._page_test()), ("AYARLAR", self._page_settings()))
        for i, (name, page) in enumerate(pages):
            b = QPushButton(name); b.setObjectName("nav"); b.setCheckable(True); b.setChecked(i == 0)
            self.nav_group.addButton(b, i); nav.addWidget(b); self.stack.addWidget(page)
        self.nav_group.idClicked.connect(self.stack.setCurrentIndex); nav.addStretch(1)
        right.addLayout(nav); right.addWidget(self.stack, 1)

        foot = panel(); fl = QHBoxLayout(foot); fl.setContentsMargins(14, 12, 14, 12); fl.setSpacing(16)
        self.listen_button = QPushButton("▶  DİNLEMEYİ BAŞLAT"); self.listen_button.setObjectName("primary")
        self.listen_button.setMinimumSize(240, 50); self.listen_button.clicked.connect(self.toggle_listening)
        col = QVBoxLayout(); col.setSpacing(2)
        self.state_label = label("● DİNLEME KAPALI", "mono"); self.status_label = label("Hazır", None, True)
        col.addWidget(self.state_label); col.addWidget(self.status_label)
        rel = QPushButton("TÜM TUŞLARI BIRAK"); rel.setObjectName("danger"); rel.clicked.connect(self.release_all)
        fl.addWidget(self.listen_button); fl.addLayout(col, 1); fl.addWidget(rel)
        outer.addWidget(foot)

    def _build_hero(self):
        f = panel(); f.setFixedWidth(360); l = QVBoxLayout(f); l.setContentsMargins(20, 18, 20, 18); l.setSpacing(10)
        l.addWidget(label("SİNYAL", "mono")); self.scope = Scope(); l.addWidget(self.scope, 1)
        l.addWidget(label("SON KOMUT", "mono"))
        self.hero_phrase = label("—", "hero"); l.addWidget(self.hero_phrase)
        row = QHBoxLayout(); self.hero_key = Keycap("", 96, 76); row.addWidget(self.hero_key)
        col = QVBoxLayout(); col.addWidget(label("GECİKME", "mono")); self.hero_latency = label("— ms", "h2"); col.addWidget(self.hero_latency)
        row.addSpacing(12); row.addLayout(col); row.addStretch(1); l.addLayout(row)
        l.addSpacing(6)
        badge = label("● CİHAZ İÇİ TANIMA  ·  KONUŞMA KAYDEDİLMEZ", "mono", True)
        badge.setStyleSheet(f"color:{AMBER}"); l.addWidget(badge)
        return f

    def _page_commands(self):
        page = QWidget(); l = QVBoxLayout(page); l.setContentsMargins(0, 6, 0, 0); l.setSpacing(12)
        self.pattern_status = label("", "mono", True); l.addWidget(self.pattern_status)
        self.list_host = QWidget(); self.list_lay = QVBoxLayout(self.list_host)
        self.list_lay.setContentsMargins(0, 0, 6, 0); self.list_lay.setSpacing(8); self.list_lay.addStretch(1)
        l.addWidget(scroll(self.list_host), 1)

        ed = panel(); el = QVBoxLayout(ed); el.setContentsMargins(16, 14, 16, 14); el.setSpacing(10)
        el.addWidget(label("KOMUT DÜZENLEYİCİ", "mono"))
        r1 = QHBoxLayout(); r1.setSpacing(10)
        self.phrase_entry = QLineEdit(); self.phrase_entry.setPlaceholderText("Söylenecek ifade (ör. zırh)")
        self.key_cap = KeyCapture()
        self.seg_press = QPushButton("KISA BASIŞ"); self.seg_hold = QPushButton("SÖYLEYİNCE BIRAK")
        grp = QButtonGroup(self); grp.setExclusive(True)
        for b in (self.seg_press, self.seg_hold): b.setObjectName("seg"); b.setCheckable(True); grp.addButton(b)
        self.seg_press.setChecked(True)
        seg = QHBoxLayout(); seg.setSpacing(0); seg.addWidget(self.seg_press); seg.addWidget(self.seg_hold)
        r1.addWidget(self.phrase_entry, 1); r1.addWidget(self.key_cap); r1.addLayout(seg); el.addLayout(r1)
        r2 = QHBoxLayout(); r2.setSpacing(10)
        save = QPushButton("KOMUTU KAYDET"); save.setObjectName("primary"); save.clicked.connect(self.upsert_command)
        wiz = QPushButton("● SÖYLEYEREK EKLE"); wiz.clicked.connect(self.toggle_wizard)
        self.wiz_button = wiz
        self.enroll_button = QPushButton("5 ÖRNEK KAYDET"); self.enroll_button.clicked.connect(self.open_enrollment)
        rm = QPushButton("SİL"); rm.setObjectName("danger"); rm.clicked.connect(self.remove_command)
        for b in (save, wiz, self.enroll_button): r2.addWidget(b)
        r2.addStretch(1); r2.addWidget(rm); el.addLayout(r2); l.addWidget(ed)
        return page

    def _page_test(self):
        page = QWidget(); l = QVBoxLayout(page); l.setContentsMargins(0, 6, 0, 0); l.setSpacing(12)
        l.addWidget(label("ÖNCE GÜVENLE DENEYİN", "h2"))
        l.addWidget(label("Bu ekranda hiçbir tuş oyuna gönderilmez. İki yöntemi aynı ifadeyle ayrı ayrı deneyin; ‘kararsız’ sonuç tuş tetiklemez.", "dim", True))
        row = QHBoxLayout(); row.setSpacing(10)
        self.expected_entry = QLineEdit(); self.expected_entry.setPlaceholderText("Hedef ifade (ör. zırh)")
        self.expected_entry.textEdited.connect(lambda _: self.stop_listening() if self._running() and self.is_test else None)
        go = QPushButton("▶  DİNLEMEYİ DENE"); go.setObjectName("primary"); go.clicked.connect(self.start_test)
        clr = QPushButton("TEMİZLE"); clr.clicked.connect(self.clear_results)
        row.addWidget(self.expected_entry, 1); row.addWidget(go); row.addWidget(clr); l.addLayout(row)
        learning = panel(); fl = QVBoxLayout(learning)
        fl.addWidget(label("SİNİR AĞINA GERİ BİLDİRİM", "mono"))
        fl.addWidget(label("Son 20 söyleyişten birini seçin. Doğru'ya basın veya gerçekte söylediğiniz komutu seçerek düzeltin. Yalnız sayısal ses özelliği öğrenilir; ham ses tutulmaz.", "dim", True))
        self.review_menu = QComboBox(); fl.addWidget(self.review_menu)
        fr = QHBoxLayout()
        self.feedback_correct = QPushButton("DOĞRU — ÖĞREN")
        self.feedback_correct.clicked.connect(lambda: self._manual_learning(False))
        fr.addWidget(self.feedback_correct)
        correction_row = QHBoxLayout()
        self.feedback_label = QComboBox(); correction_row.addWidget(self.feedback_label, 1)
        fix = QPushButton("SEÇİLEN KOMUT OLARAK ÖĞRET")
        fix.clicked.connect(lambda: self._manual_learning(True)); correction_row.addWidget(fix)
        reject = QPushButton("KOMUT DEĞİL / ÖĞRENME")
        reject.clicked.connect(self._reject_learning); fr.addWidget(reject)
        fl.addLayout(fr); fl.addLayout(correction_row)
        self.learning_status = label("Henüz sinir ağı sonucu yok. Canlı dinleme sonuçları da burada incelenebilir.", "dim", True)
        self.learning_totals = label("", "mono", True); fl.addWidget(self.learning_totals)
        fl.addWidget(self.learning_status); l.addWidget(learning)
        self.capture_validation = QCheckBox("Doğrulama özelliklerini RAM'de tut (öğrenme bu testte durur)")
        l.addWidget(self.capture_validation)
        self.capture_validation.toggled.connect(lambda _: self.stop_listening() if self._running() and self.is_test else None)
        actions=QHBoxLayout()
        summary=QPushButton("DOĞRULAMA ÖZETİ"); summary.clicked.connect(self.show_summary)
        export=QPushButton("DOĞRULAMA ÖZELLİKLERİNİ DIŞA AKTAR"); export.clicked.connect(self.export_validation)
        actions.addWidget(summary); actions.addWidget(export); l.addLayout(actions)
        self.results_host = QWidget(); self.results_lay = QVBoxLayout(self.results_host)
        self.results_lay.setContentsMargins(0, 0, 6, 0); self.results_lay.setSpacing(8); self.results_lay.addStretch(1)
        l.addWidget(scroll(self.results_host), 1)
        return page

    def _page_settings(self):
        host = QWidget(); l = QVBoxLayout(host); l.setContentsMargins(0, 6, 8, 0); l.setSpacing(12)
        g = panel(); gl = QVBoxLayout(g); gl.setContentsMargins(18, 16, 18, 16); gl.setSpacing(10)
        gl.addWidget(label("GÜVENLİK", "mono"))
        self.focus_check = QCheckBox("Yalnızca seçtiğim oyun penceresi etkinken tuş gönder (önerilen)"); self.focus_check.setChecked(True)
        gl.addWidget(self.focus_check)
        gl.addWidget(label("Başlık boşsa her pencerede çalışır. Bir kelime yazarsanız yalnızca eşleşen pencerede çalışır.", "dim", True))
        self.target_entry = QLineEdit(); self.target_entry.setPlaceholderText("Örn. ROB RUN"); gl.addWidget(self.target_entry)
        pick = QPushButton("PENCEREYİ SEÇ · 8 SANİYE İÇİNDE HEDEF PENCEREYE GEÇİN")
        pick.clicked.connect(self.begin_window_pick); gl.addWidget(pick, 0, Qt.AlignLeft)
        sv = QPushButton("GÜVENLİK AYARINI PROFİLE KAYDET"); sv.clicked.connect(lambda: self.save_profile())
        gl.addWidget(sv, 0, Qt.AlignLeft); l.addWidget(g)

        compat = panel(); cl = QVBoxLayout(compat); cl.setContentsMargins(18, 16, 18, 16)
        cl.addWidget(label("TUŞ GÖNDERİMİ", "mono"))
        self.scan_check = QCheckBox("Tarama koduyla gönder (oyunlar için önerilir)")
        self.scan_check.setChecked(True)
        cl.addWidget(self.scan_check)
        duration_row = QHBoxLayout(); duration_row.addWidget(label("Kısa basış süresi", "dim"))
        self.press_duration = QSpinBox(); self.press_duration.setRange(0, 500); self.press_duration.setValue(0)
        self.press_duration.setSuffix(" ms"); self.press_duration.setSpecialValueText("Anında")
        duration_row.addWidget(self.press_duration); duration_row.addStretch(1); cl.addLayout(duration_row)
        cl.addWidget(label("Önce tarama kodunu deneyin. Gerekirse süreyi 50–100 ms yapın; ayarlar bu oturumda geçerlidir.", "dim", True))
        l.addWidget(compat)

        s = panel(); sl = QVBoxLayout(s); sl.setContentsMargins(18, 16, 18, 16); sl.setSpacing(8)
        sl.addWidget(label("HIZ ↔ YANLIŞ ALGILAMA DENGESİ", "mono"))
        row = QHBoxLayout(); row.addWidget(label("DAHA HIZLI\nyanlış tetikleme riski", "dim"))
        self.sens = QSlider(Qt.Horizontal); self.sens.setRange(1, 100); self.sens.setValue(65)
        self.sens.valueChanged.connect(self._sens_changed); row.addWidget(self.sens, 1)
        r = label("DAHA SEÇİCİ\ndoğru sözü reddedebilir", "dim"); r.setAlignment(Qt.AlignRight); row.addWidget(r)
        sl.addLayout(row); self.sens_label = label("Dengeli ayar", "mono"); sl.addWidget(self.sens_label); l.addWidget(s)

        self.verify_sounds_check = QCheckBox("Benzer seslerde ek doğrulama (yanlış tuşları azaltır)")
        self.verify_sounds_check.setChecked(self._profile().get('verify_similar_sounds',True))
        self.verify_sounds_check.toggled.connect(self._set_sound_verification)
        l.addWidget(self.verify_sounds_check)
        l.addWidget(label("Sinir ağıyla bağımsız ses kontrolü anlaşmazsa tuş gönderilmez. Aynı eğitimden alınan ses özellikleri, yoksa bu profildeki mevcut yeni MFCC kayıtları kullanılır. Eksik kayıt varsa yeniden öğretmek gerekir. Değişiklik yeni dinlemede uygulanır.", "dim", True))
        self.auto_learning_check = QCheckBox("Sinir ağı: güçlü eşleşmelerden otomatik öğren")
        self.auto_learning_check.setChecked(self._profile().get('auto_learning', True))
        self.auto_learning_check.toggled.connect(self._set_auto_learning)
        l.addWidget(self.auto_learning_check)
        l.addWidget(label("Otomatik öğrenme doğruluğu kesin bilemez. İlk model ve güncel model anlaşırsa, üç farklı güçlü söyleyişten sonra sınırlı örnek ekler. İlk kayıtlar korunur. Deneme sekmesinden manuel onay/düzeltme yapabilirsiniz.", "dim", True))
        undo = QPushButton("SON ÖĞRENMEYİ GERİ AL (BU OTURUM)")
        undo.clicked.connect(self._undo_learning); l.addWidget(undo)
        reset = QPushButton("SONRADAN ÖĞRENİLENLERİ SİL — İLK KAYITLARI KORU")
        reset.clicked.connect(self._reset_learning); l.addWidget(reset)
        self.continuous_check = QCheckBox("Cümle içinde kişisel komut ara (deneysel)")
        self.continuous_check.setChecked(self._profile().get("continuous_spotting", False))
        self.continuous_check.toggled.connect(self._save_continuous)
        l.addWidget(self.continuous_check)
        l.addWidget(label("Açıksa komut cümle içinde geçince de çalışabilir; cümlenin bitmesini beklemez. Normal konuşmadaki aynı sözcük de tuşu tetikleyebilir. Önce Deneme'de sınayın. Sessizlik ve erken karar ayarları yalnız kapalıyken kullanılır.", "dim", True))
        self.decision_menu = QComboBox()
        self.decision_menu.addItems(["Seçici", "Dengeli", "Atak"])
        self.decision_menu.setCurrentIndex(["selective", "balanced", "aggressive"].index(self._profile().get("decision_mode", "balanced")))
        l.addWidget(label("KİŞİSEL ÖRÜNTÜ · KARAR MODU", "mono")); l.addWidget(self.decision_menu)
        l.addWidget(label("Seçici: yanlış tuştan kaçınır. Dengeli: önerilen. Atak: daha kolay kabul eder, yanlış tuş riski artar.", "dim", True))
        self.decision_menu.currentIndexChanged.connect(self._save_decision_mode)
        verify = QPushButton("ERKEN KARARI DENEME SONUÇLARIYLA ETKİNLEŞTİR")
        verify.clicked.connect(self._enable_early_decisions); l.addWidget(verify)
        l.addWidget(label("Varsayılan kapalı. Bu oturumda en az 5 doğru ve sıfır hatalı/reddedilen denemesi olan komutlar için açılır. Yeniden öğretmede kapanır.", "dim", True))
        disable = QPushButton("ERKEN KARARI KAPAT")
        disable.clicked.connect(self._disable_early_decisions); l.addWidget(disable)
        timing = panel(); tl = QVBoxLayout(timing); tl.setContentsMargins(18, 16, 18, 16)
        tl.addWidget(label("KİŞİSEL ÖRÜNTÜ · SESSİZLİK BEKLEMESİ", "mono"))
        self.silence_duration = QSpinBox(); self.silence_duration.setRange(100, 600); self.silence_duration.setSingleStep(25)
        self.silence_duration.setValue(self._profile().get("silence_ms", 150)); self.silence_duration.setSuffix(" ms"); tl.addWidget(self.silence_duration)
        self.silence_duration.setKeyboardTracking(False)
        self.silence_duration.valueChanged.connect(self._save_silence_duration)
        tl.addWidget(label("100–600 ms: düşük değer daha hızlı karar verir; sözü erken kesebilir. Değişiklik bu oyun profiline otomatik kaydedilir. Durdurup yeniden dinlemeye başlayınca uygulanır.", "dim", True)); l.addWidget(timing)

        p = panel(); pl = QVBoxLayout(p); pl.setContentsMargins(18, 16, 18, 16); pl.setSpacing(8)
        pl.addWidget(label("GİZLİLİK", "mono"))
        pl.addWidget(label("Ham ses kaydedilmez. Onayınızla yalnızca sayısal ses özellikleri bu profile yazılır; istediğiniz zaman silebilirsiniz.", "dim", True))
        d = QPushButton("BU PROFİLİN SES ÖZELLİKLERİNİ SİL"); d.setObjectName("danger"); d.clicked.connect(self.delete_audio_templates)
        pl.addWidget(d, 0, Qt.AlignLeft); l.addWidget(p)

        sysf = panel(); yl = QVBoxLayout(sysf); yl.setContentsMargins(18, 16, 18, 16); yl.setSpacing(8)
        yl.addWidget(label("SİSTEM", "mono")); self.resource_label = label("CPU/RAM: ölçüm bekleniyor", "dim"); yl.addWidget(self.resource_label)
        br = QHBoxLayout(); rf = QPushButton("MİKROFONLARI YENİLE"); rf.clicked.connect(self.refresh_devices)
        self.dl_btn = QPushButton("TÜRKÇE TANIMA MODELİNİ İNDİR"); self.dl_btn.clicked.connect(self.download_model)
        br.addWidget(rf); br.addWidget(self.dl_btn); br.addStretch(1); yl.addLayout(br); l.addWidget(sysf)
        l.addStretch(1)
        return scroll(host)

    # ---------- state ----------
    def _profile(self): return self.profiles[self.profile_name]
    def _method(self):
        return next((key for key, value in METHODS.items() if value == self.method_menu.currentText()), 'neural')

    def _template_field(self):
        return 'neural_templates' if self._method() == 'neural' else 'audio_templates'

    def _pending_field(self):
        return 'pending_neural_templates' if self._method() == 'neural' else 'pending_audio_templates'

    def _method_changed(self, *_):
        self._clear_learning_review()
        self.log.clear(); self._pending_record=None
        if self._running(): self.stop_listening()
        self.validation_counts.clear()
        self.refresh_commands(); self.update_pattern_status()
        self.set_status('Sinir ağı için yeni örnekler gerekir; eski MFCC kayıtları dönüştürülemez.' if self._method() == 'neural' else 'Tanıma yöntemi değiştirildi.')
    def _running(self): return bool(self.worker and self.worker.isRunning())
    def info(self, title, msg): QMessageBox.information(self, title, msg)
    def set_status(self, msg): self.status_label.setText(msg)

    def _set_state(self, key):
        text, mode = self.STATES[key]
        self.state_label.setText("● " + text); self.scope.set_mode(mode)
        self.state_label.setStyleSheet(f"color:{DIM if key == 'off' else AMBER}")
        self.listen_button.setText("▶  DİNLEMEYİ BAŞLAT" if key == "off" else "■  DURDUR")

    def _change_profile(self, name):
        self._clear_learning_review()
        if name not in self.profiles: return
        self.log.clear(); self._pending_record=None
        if self._running(): self.stop_listening()
        self.validation_counts.clear()
        self.profile_name = name; self.selected = ""
        self.target_entry.setText(self._profile().get("target_window", ""))
        self.verify_sounds_check.blockSignals(True)
        self.verify_sounds_check.setChecked(self._profile().get('verify_similar_sounds',True))
        self.verify_sounds_check.blockSignals(False)
        self.auto_learning_check.blockSignals(True)
        self.auto_learning_check.setChecked(self._profile().get('auto_learning', True))
        self.auto_learning_check.blockSignals(False)
        self.continuous_check.blockSignals(True)
        self.continuous_check.setChecked(self._profile().get("continuous_spotting", False))
        self.continuous_check.blockSignals(False)
        self.decision_menu.blockSignals(True)
        self.decision_menu.setCurrentIndex(["selective", "balanced", "aggressive"].index(self._profile().get("decision_mode", "balanced")))
        self.decision_menu.blockSignals(False)
        self.silence_duration.blockSignals(True)
        self.silence_duration.setValue(self._profile().get("silence_ms", 150))
        self.silence_duration.blockSignals(False)
        self.refresh_commands(); self.update_pattern_status()

    def _save_continuous(self, enabled):
        self._profile()["continuous_spotting"] = enabled
        self.validation_counts.clear()
        self.save_profile(silent=True)
        self.set_status("Komut arama ayarı kaydedildi; yeni dinlemede uygulanır.")

    def _enable_early_decisions(self):
        self._profile()["early_commands"] = [k for k, v in self.validation_counts.items() if v[0] >= 5 and v[1:] == [0, 0] and k in self._profile()["commands"]]
        self.save_profile(silent=True)
        self.set_status("Erken karar yeni dinlemede uygulanır: " + ", ".join(self._profile()["early_commands"]))

    def _disable_early_decisions(self):
        self._profile()["early_commands"] = []; self.save_profile(silent=True)
        self.set_status("Erken karar kapatıldı; dinlemeyi yeniden başlatın.")

    def _save_decision_mode(self, index):
        self.validation_counts.clear()
        self._profile()["early_commands"] = []
        self._profile()["decision_mode"] = ["selective", "balanced", "aggressive"][index]
        self.save_profile(silent=True)

    def _save_silence_duration(self, value):
        self._profile()["silence_ms"] = value
        if self.save_profile(silent=True):
            self.set_status(f"Sessizlik beklemesi {value} ms olarak profile kaydedildi. Yeni dinlemede uygulanır.")

    def refresh_devices(self):
        try:
            import sounddevice as sd
            labels = {"Varsayılan mikrofon": None}
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"]: labels[f"{i}: {d['name']}"] = i
            self.device_labels = labels
            cur = self.device_menu.currentText()
            self.device_menu.blockSignals(True); self.device_menu.clear(); self.device_menu.addItems(list(labels))
            self.device_menu.setCurrentText(cur if cur in labels else "Varsayılan mikrofon"); self.device_menu.blockSignals(False)
            self.device_index = labels[self.device_menu.currentText()]
            self.set_status(f"{len(labels) - 1} mikrofon bulundu.")
        except Exception as exc:
            self.set_status(f"Mikrofon listesi alınamadı: {exc}")

    def refresh_commands(self):
        self._update_learning_totals()
        while self.list_lay.count() > 1:
            w = self.list_lay.takeAt(0).widget()
            if w: w.deleteLater()
        cmds, tpl = self._profile().get("commands", {}), self._profile().get(self._template_field(), {})
        for i, (phrase, a) in enumerate(cmds.items()):
            pending = self._profile().get(self._pending_field(), {})
            state = ("YENİ ÖRNEKLER KAYDEDİLDİ · 5/5 · ETKİNLEŞMEYİ BEKLİYOR" if phrase in pending else
                     "YENİ ÖRÜNTÜ ETKİN · 5/5" if tpl.get(phrase, {}).get("version") in (2, 3) else
                     "ESKİ ÖRÜNTÜ ETKİN · YENİDEN ÖĞRETİN" if phrase in tpl else False)
            row = CommandRow(phrase, a["key"], a["mode"], state, phrase == self.selected)
            row.clicked.connect(self.select_command); self.list_lay.insertWidget(i, row)
        if not cmds:
            self.list_lay.insertWidget(0, label("Henüz komut yok. ‘Söyleyerek ekle’ ile başlayın.", "dim"))

    def select_command(self, phrase):
        a = self._profile()["commands"].get(phrase)
        if not a: return
        self.selected = phrase; self.phrase_entry.setText(phrase)
        self.key_cap.set_key(a["key"], a.get("vk", KeyController.vk(a["key"])))
        (self.seg_press if a["mode"] == "press" else self.seg_hold).setChecked(True)
        self.enroll_button.setText("5 ÖRNEK KAYDET")
        self.refresh_commands(); self.update_pattern_status()

    def update_pattern_status(self):
        cmds, tpl = self._profile().get("commands", {}), self._profile().get(self._template_field(), {})
        pending = self._profile().get(self._pending_field(), {})
        p = self.selected
        if pending:
            remaining = sorted(set(cmds) - set(pending))
            n = len(pending.get(p, {}).get("samples", []))
            self.pattern_status.setText(
                f"YENİ EĞİTİM: {len(set(cmds) & set(pending))}/{len(cmds)} KOMUT KAYDEDİLDİ"
                f" · SEÇİLİ YENİ ÖRNEK {n}/5\n"
                + ("Bu komutun yeni örnekleri diske kaydedildi. " if p in pending else "")
                + "Yeni eğitimi etkinleştirmek için kalanlar: " + ", ".join(remaining)
                + "\nTamamlanana kadar mevcut örüntüler kullanılır.")
        else:
            ready = sum(p in tpl for p in cmds)
            n = len(tpl.get(p, {}).get("samples", [])) if p else 0
            nxt = ("Başlangıç: bir komut ekleyin." if not cmds else
                   f"Sıradaki: Deneme sekmesinde ‘{p}’ ifadesini kişisel yöntemle sınayın." if p in tpl else
                   f"Sıradaki: ‘{p}’ için ‘5 örnek kaydet’e basın." if p else "Başlamak için bir komuta tıklayın.")
            self.pattern_status.setText(f"KİŞİSEL SES ÖRÜNTÜSÜ {ready}/{len(cmds)} HAZIR · SEÇİLİ ÖRNEK {n}/5\n{nxt}")

    # ---------- commands ----------
    def upsert_command(self):
        phrase, key = normalize(self.phrase_entry.text()), self.key_cap.key
        if not phrase or not key:
            self.info("Bilgi eksik", "İfadeyi yazın ve ‘Tuşu yakala’ ile bir tuş seçin."); return
        vk = self.key_cap.vk if self.key_cap.vk is not None else KeyController.vk(key)
        self._profile()["commands"][phrase] = {"key": key, "vk": vk, "mode": "press" if self.seg_press.isChecked() else "hold"}
        self.save_profile(silent=True); self.selected = phrase; self.refresh_commands(); self.update_pattern_status()
        self.set_status(f"‘{phrase}’ komutu profile kaydedildi.")

    def remove_command(self):
        p = self.selected
        if not p: self.info("Önce komut seçin", "Silmek istediğiniz komuta tıklayın."); return
        if QMessageBox.question(self, "Komutu sil", f"‘{p}’ komutu ve varsa kişisel örüntüsü silinsin mi?") != QMessageBox.Yes: return
        for field in ("pending_audio_templates", "pending_neural_templates", "neural_templates"):
            self._profile().get(field, {}).pop(p, None)
        self.validation_counts.clear(); self._profile()["early_commands"] = []
        self._profile()["commands"].pop(p, None); self._profile().get("audio_templates", {}).pop(p, None)
        self.selected = ""; self.save_profile(silent=True); self.refresh_commands(); self.update_pattern_status()

    def add_profile(self):
        name, ok = QInputDialog.getText(self, "Profil ekle", "Yeni oyun profilinin adı:")
        name = name.strip()
        if not ok or not name: return
        if name in self.profiles: self.info("Profil zaten var", "Bu adla bir profil bulunuyor."); return
        self.profiles[name] = {"name": name, "commands": {}, "target_window": "", "audio_templates": {}}
        self.profile_menu.blockSignals(True); self.profile_menu.addItem(name); self.profile_menu.setCurrentText(name)
        self.profile_menu.blockSignals(False); self._change_profile(name); self.save_profile(silent=True)

    def save_profile(self, silent=False):
        self._profile()["target_window"] = self.target_entry.text().strip()
        try:
            save_profiles(PROFILE_FILE, self.profiles)
            if not silent: self.set_status("Profil ve güvenlik ayarları kaydedildi.")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Kaydedilemedi", str(exc)); return False

    def delete_audio_templates(self):
        tpl = self._profile().get(self._template_field(), {})
        if not tpl: self.info("Henüz örüntü yok", "Bu profilde silinecek kişisel ses özelliği yok."); return
        if QMessageBox.question(self, "Ses özelliklerini sil", f"{len(tpl)} kişisel örüntü silinsin mi?") != QMessageBox.Yes: return
        self._profile()[self._pending_field()] = {}; self._profile()["early_commands"] = []
        self._profile()[self._template_field()] = {}; self.save_profile(silent=True)
        self.refresh_commands(); self.update_pattern_status(); self.set_status("Kişisel ses özellikleri silindi.")

    # ---------- enrollment ----------
    def open_enrollment(self):
        p = self.selected
        if not p: self.info("Bir komut seçin", "Önce listeden komuta tıklayın."); return
        if self._running(): self.info("Dinleme açık", "Örnek kaydetmeden önce dinlemeyi durdurun."); return
        self.method_menu.setEnabled(False); self.profile_menu.setEnabled(False)
        d = QDialog(self); d.setWindowTitle("Kişisel ses örneği"); d.setMinimumWidth(520)
        l = QVBoxLayout(d); l.setContentsMargins(26, 22, 26, 22); l.setSpacing(12)
        l.addWidget(label("SES ÖRNEĞİ KAYDI", "mono")); l.addWidget(label(f"‘{p}’", "hero"))
        l.addWidget(label("Başlat’a basın, sonra ifadeyi 5 kez söyleyin. Her söyleyişten sonra kısa duraklayın. Ham ses saklanmaz; sayısal özellikler ses karakteristiği içerebilir ve istediğiniz zaman silinebilir.", "dim", True))
        consent = QCheckBox("Bu profile kişisel ses özellikleri kaydetmeyi onaylıyorum."); l.addWidget(consent)
        pr = QHBoxLayout(); pips = []
        for _ in range(5):
            pip = label("●"); pip.setStyleSheet("color:#3a342c;font-size:26px"); pips.append(pip); pr.addWidget(pip)
        pr.addStretch(1); l.addLayout(pr)
        msg = label("Henüz kayıt başlamadı.", "mono", True); l.addWidget(msg)
        br = QHBoxLayout(); start = QPushButton("● KAYDI BAŞLAT"); start.setObjectName("primary"); start.setEnabled(False)
        done = QPushButton("BİTTİ — DENEMEYE GEÇ"); done.setEnabled(False); close = QPushButton("KAPAT")
        br.addWidget(start); br.addWidget(done); br.addStretch(1); br.addWidget(close); l.addLayout(br)
        consent.toggled.connect(start.setEnabled)

        def begin():
            start.setEnabled(False); msg.setText("0/5 alındı — şimdi ifadeyi söyleyin.")
            w = PatternRecognitionWorker(self.device_index, {}, self.sens.value() / 100, enrollment_phrase=p, sample_count=5, silence_seconds=self.silence_duration.value()/1000, enrollment_version=3 if self._method() == "neural" else 2)
            self._start_worker(w, "enroll"); self._set_state("enroll")
        def finish():
            pending = self._profile().get(self._pending_field(), {})
            if pending:
                remaining = sorted(set(self._profile()["commands"]) - set(pending))
                d.accept()
                self.stack.setCurrentIndex(0); self.nav_group.button(0).setChecked(True)
                if remaining: self.select_command(remaining[0])
                return
            self.stack.setCurrentIndex(1); self.nav_group.button(1).setChecked(True)
            self.expected_entry.setText(p); self.method_menu.setCurrentText(METHODS["neural"]); d.accept()
        start.clicked.connect(begin); done.clicked.connect(finish); close.clicked.connect(d.reject)
        d.finished.connect(self._enroll_closed)
        self.enr = {"dialog": d, "msg": msg, "pips": pips, "start": start, "done": done, "phrase": p}
        d.open()

    def _enroll_closed(self, *_):
        self.method_menu.setEnabled(True); self.profile_menu.setEnabled(True)
        if self.worker_kind == "enroll" and self._running(): self.stop_listening()
        self.enr = None

    def _on_progress(self, done, total):
        if not self.enr: return
        for i, pip in enumerate(self.enr["pips"]):
            pip.setStyleSheet(f"color:{AMBER if i < done else '#3a342c'};font-size:26px")
        self.enr["msg"].setText(f"{done}/{total} örnek alındı — her seferinde kısa duraklayın.")

    def _on_trained(self, phrase, template):
        if not self.enr: return
        try:
            self.validation_counts.clear()
            self._clear_learning_review()
            self._profile()["early_commands"] = []
            pending_field = 'pending_neural_templates' if template['version'] == 3 else 'pending_audio_templates'
            active_field = 'neural_templates' if template['version'] == 3 else 'audio_templates'
            pending = self._profile().setdefault(pending_field, {})
            pending[phrase] = template
            required = set(self._profile()["commands"])
            if required <= set(pending):
                self._profile()[active_field] = {key: pending[key] for key in required}
                self._profile()[pending_field] = {}
                if template["version"] == 3: self._profile()["adaptive_learning"] = {}
            save_profiles(PROFILE_FILE, self.profiles)
            self.refresh_commands(); self.update_pattern_status(); self._on_progress(5, 5)
            remaining = required - set(pending)
            message = ("Yeni eğitim hazır. Deneme turunda her komutu en az 5 kez söyleyin." if not self._profile().get(pending_field) else
                       "Kayıt alındı. Yeni yöntemi açmak için kalan komutları öğretin: " + ", ".join(sorted(remaining)))
            if not self._profile().get(pending_field):
                from .acoustic import ShortCommandMatcher
                if template['version'] == 3:
                    from .neural import NeuralMatcher
                    calibrated = NeuralMatcher(self._profile()[active_field])
                else: calibrated = ShortCommandMatcher(self._profile()[active_field])
                if calibrated.warnings: message += " Karışma riski: " + ", ".join(calibrated.warnings)
            self.enr["done"].setText("BİTTİ — SIRADAKİ KOMUTA GEÇ" if self._profile().get(pending_field) else "BİTTİ — DENEMEYE GEÇ")
            self.enr["msg"].setText(message)
            self.enr["done"].setEnabled(True); self._set_state("off")
            self.set_status(f"‘{phrase}’ için ses örüntüsü kaydedildi.")
        except Exception as exc:
            self.set_status(f"Ses özellikleri kaydedilemedi: {exc}")

    # ---------- listening ----------
    def _start_worker(self, w, kind):
        self.worker, self.worker_kind = w, kind
        self._test_expected=normalize(self.expected_entry.text()) if self.is_test else None
        w.status.connect(self._on_status); w.failed.connect(self._on_error)
        w.phrase.connect(lambda *args: self._on_phrase(*args) if self.worker is w and not w._stop.is_set() else None); w.observed.connect(self._on_observed)
        if hasattr(w, 'learning_candidate'):
            from .learning import training_id
            profile_name = self.profile_name
            base_id = training_id(self._profile().get('neural_templates', {}))
            w.learning_candidate.connect(lambda event: self._on_learning_candidate(profile_name, base_id, event) if self.worker is w else None)
        self._pending_record = None
        if hasattr(w,'record'):
            captured_profile = self.profile_name
            captured_method = self._method()
            w.record.connect(lambda rec: self._on_record(rec) if self.worker is w and self.profile_name==captured_profile and self._method()==captured_method and not w._stop.is_set() else None)
        w.level.connect(self.scope.amp)
        self.hero_latency.setToolTip("Konuşma başlangıcından tanıma kararına kadar geçen süre; oyunun tepki süresini içermez.")
        if hasattr(w, "diagnostics"): w.diagnostics.connect(self._on_diagnostics)
        if hasattr(w, "timings"): w.timings.connect(self._on_timings)
        if hasattr(w, "progress"): w.progress.connect(self._on_progress)
        if hasattr(w, "trained"): w.trained.connect(self._on_trained)
        w.start()

    def _on_status(self, text):
        self.set_status(text)
        if self.worker_kind == "enroll" and self.enr and "otomatik kaydediliyor" not in text:
            self.enr["msg"].setText(text)

    def toggle_listening(self):
        if self._running(): self.stop_listening()
        else: self.start_listening(False)

    def start_test(self):
        if not normalize(self.expected_entry.text()):
            self.info("Hedef ifade gerekli", "‘Hedef ifade’ kutusuna söyleyeceğiniz kelimeyi yazın."); return
        if self._method() in ("pattern", "neural"):
            missing = sorted(set(self._profile().get("commands", {})) - set(self._profile().get(self._template_field(), {})))
            if missing: self.info("Ses örneği eksik", "Her komut için 5 örnek kaydedin. Eksik: " + ", ".join(missing)); return
        self.start_listening(True)

    def start_listening(self, test=False):
        if self._running(): self.stop_listening()
        method, sens = self._method(), self.sens.value() / 100
        if method == "vosk":
            if not MODEL.is_dir(): self.info("Türkçe model gerekli", "Ayarlar’daki ‘Türkçe tanıma modelini indir’ düğmesini kullanın."); return
            w = RecognitionWorker(MODEL, self.device_index, sens, commands=self._profile().get("commands", {}), test_mode=test)
        else:
            tpl = self._profile().get(self._template_field(), {})
            if not tpl: self.info("Önce örnek kaydedin", "Bir komut seçip 5 kez ses örneği kaydedin."); return
            from .verification import verification_templates
            verify = verification_templates(tpl,self._profile().get('audio_templates',{})) if method == 'neural' and self._profile().get('verify_similar_sounds',True) else {}
            w = PatternRecognitionWorker(self.device_index, tpl, sens, phonetic_templates=verify, test_mode=test, silence_seconds=self.silence_duration.value()/1000, decision_mode=self._profile().get("decision_mode", "balanced"), early_commands=self._profile().get("early_commands", []), continuous=self._profile().get("continuous_spotting", False), learning=self._profile().get("adaptive_learning", {}), capture_features=self.capture_validation.isChecked() and test)
        self.is_test = test; self.save_profile(silent=True)
        self._start_worker(w, "test" if test else method); self._set_state("test" if test else "live")

    def stop_listening(self):
        if self._running(): self.worker.stop(); self.worker.wait(2000)
        self._set_state("off"); self.wiz_button.setText("● SÖYLEYEREK EKLE")
        if self.worker_kind == "enroll" and self.enr:
            self.enr["msg"].setText("Kayıt durduruldu. Yeniden başlatabilir veya kapatabilirsiniz.")
            self.enr["start"].setEnabled(True)
        self.release_all()

    def _on_error(self, message):
        self.stop_listening(); self.set_status(f"Mikrofon veya tanıma hatası: {message}")
        if self.enr: self.enr["msg"].setText(f"Mikrofon açılamadı.\n{message}")
        else: QMessageBox.critical(self, "Mikrofon açılamadı", f"Mikrofon seçimini ve Windows mikrofon iznini kontrol edin.\n\n{message}")

    def toggle_wizard(self):
        if self.worker_kind == "wizard" and self._running(): self.stop_listening(); return
        if self._running(): self.info("Önce dinlemeyi durdurun", "Komut eklemek için mevcut dinlemeyi durdurun."); return
        if not MODEL.is_dir(): self.info("Türkçe model gerekli", "Ayarlar’dan modeli indirin."); return
        self.is_test = True
        self._start_worker(RecognitionWorker(MODEL, self.device_index, self.sens.value() / 100, commands=None, test_mode=True), "wizard")
        self._set_state("wizard"); self.wiz_button.setText("■ DİNLİYOR… İPTAL")
        self.set_status("Komutu söyleyin; duyulan ifade kutuya yazılır. Sonra ‘Tuşu yakala’ya basıp bir tuşa dokunun.")

    # ---------- events ----------
    def _clear_learning_review(self):
        from .learning import AutoLearningGate
        self.review_events.clear()
        self.auto_gate = AutoLearningGate()
        self.review_menu.clear(); self.feedback_label.clear()
        self.learning_status.setText("Yeni profil/yöntem için öğrenme incelemesi temizlendi.")

    def _update_learning_totals(self):
        examples = self._profile().get('adaptive_learning', {}).get('examples', [])
        manual = sum(e['source']=='manual' for e in examples)
        self.learning_totals.setText(f"KALICI EK ÖRNEK: {manual} MANUEL · {len(examples)-manual} OTOMATİK · İLK KAYITLAR KORUNUR")
        blocked = self._profile().get('adaptive_learning',{}).get('blocked_labels',[])
        if blocked: self.learning_totals.setText(self.learning_totals.text()+"\nOTOMATİK ÖĞRENMESİ DURAKLATILAN: "+", ".join(blocked))

    def _set_sound_verification(self, enabled):
        self._profile()['verify_similar_sounds'] = enabled
        self.save_profile(silent=True)
        self.set_status("Ek ses doğrulaması kaydedildi; dinlemeyi yeniden başlatın.")

    def _set_auto_learning(self, enabled):
        before = self._profile().get('auto_learning', True)
        self._profile()['auto_learning'] = enabled
        from .learning import AutoLearningGate
        self.auto_gate = AutoLearningGate()
        if self.save_profile(silent=True):
            self.learning_status.setText("Otomatik öğrenme açık." if enabled else "Otomatik öğrenme kapalı; manuel onay kullanılabilir.")
        else:
            self._profile()['auto_learning'] = before
            self.auto_learning_check.blockSignals(True)
            self.auto_learning_check.setChecked(before)
            self.auto_learning_check.blockSignals(False)

    def _on_learning_candidate(self, profile_name, base_id, event):
        from .learning import training_id, add_example
        templates = self._profile().get('neural_templates', {})
        if self.profile_name != profile_name or self._method() != 'neural' or training_id(templates) != base_id:
            return
        event = copy.deepcopy(event)
        if self.is_test and event.get('prediction') != normalize(self.expected_entry.text()):
            event['auto_eligible'] = False
        event['base_id'] = base_id; event['received_at'] = time.monotonic()
        self.review_events[event['id']] = event
        while len(self.review_events) > 20:
            oldest = next(iter(self.review_events))
            del self.review_events[oldest]
            index = self.review_menu.findData(oldest)
            if index >= 0: self.review_menu.removeItem(index)
        self.review_menu.insertItem(0, f"{event['prediction'] or 'Reddedildi'} · {event['reason']} · {event['id'][:6]}", event['id'])
        self.review_menu.setCurrentIndex(0)
        previous = self.feedback_label.currentText()
        self.feedback_label.clear(); self.feedback_label.addItems(list(templates))
        self.feedback_label.setCurrentText(event['prediction'] or previous or event.get('suggestion',''))
        self.learning_status.setText("Son söyleyiş incelemeye hazır. Doğruysa onaylayabilir veya doğru komutu seçebilirsiniz.")
        learning = self._profile().get('adaptive_learning', {})
        active = self.worker and not self.worker._stop.is_set()
        if self._profile().get('auto_learning', True) and active and not (self.is_test and self.capture_validation.isChecked()) and self.auto_gate.consider(event, learning):
            updated = add_example(learning, templates, event, event['prediction'], 'auto')
            if self._commit_learning(updated):
                self.learning_status.setText(f"‘{event['prediction']}’ güçlü eşleşmelerden otomatik öğrenildi. Yanlışsa aşağıdan düzeltin veya geri alın.")

    def _selected_learning_event(self):
        from .learning import training_id
        event = self.review_events.get(self.review_menu.currentData())
        if (not event or time.monotonic()-event['received_at'] > 300 or
                event['base_id'] != training_id(self._profile().get('neural_templates', {}))):
            self.learning_status.setText("Bu söyleyiş yok veya süresi doldu; yeniden söyleyin.")
            return None
        return event

    def _commit_learning(self, data, remember=True):
        from .learning import preserves_enrollment, training_id, validate_learning
        templates = self._profile().get('neural_templates', {})
        try:
            data = validate_learning(data, templates)
            if data and not preserves_enrollment(templates, data, self._profile().get('decision_mode','balanced')):
                self.learning_status.setText("Bu örnek ilk kayıtların tanınmasını bozdu; eklenmedi. Yeni bir söyleyiş deneyin.")
                return False
            previous = copy.deepcopy(self._profile().get('adaptive_learning', {}))
            if data == previous:
                self.learning_status.setText("Bu söyleyiş zaten aynı şekilde öğrenilmiş.")
                return False
            profiles = copy.deepcopy(self.profiles)
            profiles[self.profile_name]['adaptive_learning'] = data
            save_profiles(PROFILE_FILE, profiles)
        except Exception as exc:
            self.learning_status.setText(f"Öğrenme kaydedilemedi: {exc}")
            return False
        self.profiles = profiles
        self._update_learning_totals()
        if remember:
            self.learning_undo.append((self.profile_name,training_id(templates),previous))
            del self.learning_undo[:-5]
        if self.worker_kind == 'neural' or (self.worker_kind == 'test' and self._method()=='neural'):
            if self.worker and hasattr(self.worker,'update_learning'): self.worker.update_learning(data)
        return True

    def _manual_learning(self, correction):
        if self.is_test and self._running() and self.capture_validation.isChecked():
            self.learning_status.setText("Doğrulama sırasında öğrenme durur; öğrenmek için önce testi durdurun.")
            return
        from .learning import add_example
        event = self._selected_learning_event()
        if not event: return
        label = self.feedback_label.currentText() if correction else event.get('prediction')
        if not label:
            self.learning_status.setText("Reddedilen söyleyiş için doğru komutu seçip ‘Seçilen komut olarak öğret’ düğmesini kullanın.")
            return
        try:
            updated = add_example(self._profile().get('adaptive_learning', {}),
                                  self._profile()['neural_templates'],event,label,'manual')
            if event.get('prediction') and label != event['prediction']:
                updated['blocked_labels'] = list(set(updated.get('blocked_labels',[])) | {event['prediction']})
        except ValueError as exc:
            self.learning_status.setText(str(exc)); return
        if self._commit_learning(updated):
            self.auto_gate.support.clear()
            self.learning_status.setText(f"‘{label}’ manuel onayınızla öğrenildi. Yeni kişisel başlık bir sonraki ses bloğunda uygulanır.")

    def _reject_learning(self):
        event = self._selected_learning_event()
        if not event: return
        self.auto_gate.support.clear()
        data = copy.deepcopy(self._profile().get('adaptive_learning', {}))
        if event.get('prediction'):
            from .learning import training_id
            from .neural import MODEL_ID
            if not data: data = {'model_id':MODEL_ID,'training_id':training_id(self._profile()['neural_templates']),'examples':[]}
            data['blocked_labels'] = list(set(data.get('blocked_labels',[])) | {event['prediction']})
        if data:
            data['examples'] = [e for e in data['examples'] if e['id'] != event['id']]
            if not self._commit_learning(data): return
        self.learning_status.setText("Bu söyleyiş öğrenmeden çıkarıldı; komut değil olarak bildirildi.")

    def _undo_learning(self):
        from .learning import training_id
        fingerprint = training_id(self._profile().get('neural_templates', {}))
        for i in range(len(self.learning_undo)-1,-1,-1):
            profile, base, data = self.learning_undo[i]
            if profile == self.profile_name and base == fingerprint:
                if self._commit_learning(data, remember=False):
                    self.learning_undo.pop(i)
                    self.learning_status.setText("Son öğrenme geri alındı.")
                return
        self.learning_status.setText("Bu eğitim için oturumda geri alınacak öğrenme yok.")

    def _reset_learning(self):
        if self._commit_learning({}):
            self._clear_learning_review()
            self.learning_status.setText("Sonradan öğrenilenler temizlendi; ilk beş kayıt korundu.")

    def _on_record(self, record):
        if self.is_test:
            target=self._test_expected if self._test_expected is not None else normalize(self.expected_entry.text())
            record.expected='' if target=='-' else target
        if not record.accepted: record.outcome=dg.OUT_NOT_ACCEPTED
        self.log.add(record)
        self._pending_record=record if record.accepted else None

    def _tag_outcome(self, outcome):
        record=self._pending_record
        if record:
            record.outcome=outcome
            if outcome==dg.OUT_KEY_SENT: record.key_sent_wall=self.last_key_sent_at
        self._pending_record=None

    def show_summary(self):
        QMessageBox.information(self,'Doğrulama özeti',self.log.format_summary()+
            "\n\nSayılar bu oturumdaki kişisel tanıma denemeleridir. Etiketsiz canlı sonuçlar doğruluk sayılmaz.\n"
            "Komut dışı konuşma testi için hedefe - yazın. Donanım gecikmesi ve oyunun tepki süresi ölçülmez.")

    def export_validation(self):
        from PySide6.QtWidgets import QFileDialog
        import json
        rows=self.log.export_features()
        if not rows:
            self.info('Doğrulama kaydı yok','Özellik toplama kutusunu işaretleyip yeni söyleyişlerle Deneme yapın. Komut dışı konuşma için hedefe - yazın.'); return
        path,_=QFileDialog.getSaveFileName(self,'Sayısal doğrulama özelliklerini dışa aktar','autopress_validation.json','JSON (*.json)')
        if path:
            try: Path(path).write_text(json.dumps({'schema_version':1,'profile':self.profile_name,'records':rows},ensure_ascii=False,allow_nan=False),encoding='utf-8')
            except (ValueError,OSError) as exc: self.info('Dışa aktarılamadı',str(exc))

    def _on_diagnostics(self, data):
        text = self.hero_latency.toolTip()
        if self.last_key_sent_at is not None and self.last_key_sent_at >= data["decision_at"]:
            text += f"\nKarardan tuşa: {(self.last_key_sent_at-data['decision_at'])*1000:.1f} ms"
            text += f"\nKonuşma sonundan tuşa: {(self.last_key_sent_at-data['speech_end'])*1000:.1f} ms"
        text += "\nKarar: " + data['reason']
        text += "\nEn yakın adaylar: " + str(data['scores'])
        if data.get("phonetic_candidate"): text += "\nEk ses kontrolü: " + data["phonetic_candidate"]
        self.hero_latency.setToolTip(text)

    def _on_timings(self, features_ms, match_ms, silence_ms):
        self.hero_latency.setToolTip(
            f"Ses özellikleri: {features_ms:.1f} ms\n"
            f"Komut karşılaştırma: {match_ms:.1f} ms\n"
            f"Konuşma sonunu bekleme: {silence_ms:.0f} ms\n"
            "Ana süre konuşmayı da içerir; oyunun tepki süresini içermez.")

    def _on_phrase(self, phrase, latency):
        if self._pending_record and self._pending_record.accepted != normalize(phrase): self._pending_record=None
        if self.worker_kind == "wizard":
            self.stop_listening(); self.phrase_entry.setText(normalize(phrase)); self.selected = ""
            self.scope.trigger(); self.hero_phrase.setText(normalize(phrase)); self.key_cap.set_key("", None)
            self.set_status("İfadeyi kontrol edin, ‘Tuşu yakala’ ile tuşu seçip kaydedin."); return
        cmd = self._profile().get("commands", {}).get(normalize(phrase))
        self.hero_phrase.setText(phrase); self.hero_latency.setText(f"{latency:.0f} ms")
        if self.is_test: self._tag_outcome(dg.OUT_TEST_ONLY); return
        if not cmd: self._tag_outcome(dg.OUT_NO_MAPPING); self.set_status(f"‘{phrase}’ duyuldu ama bu profilde tuş eşlemesi yok."); return
        if self.focus_check.isChecked() and not self._game_focused():
            self._tag_outcome(dg.OUT_BLOCKED_FOCUS)
            t = self._profile().get("target_window", "")
            why = f"‘{t}’ penceresi ön planda değil." if t else "Ayarlar’da oyun penceresi adı kayıtlı değil."
            self.set_status(f"‘{phrase}’ duyuldu; güvenlik nedeniyle tuş gönderilmedi. {why}"); return
        now, key = time.monotonic(), cmd["key"]
        if cmd["mode"] == "hold" and key in self.keys.held:
            try: self.keys.release(key)
            except Exception as exc:
                self._tag_outcome(dg.OUT_SEND_ERROR); self.set_status(f"Tuş bırakılamadı: {exc}"); return
        else:
            try:
                if cmd["mode"] == "press":
                    if key in self.keys.held: self._tag_outcome(dg.OUT_BLOCKED_HELD); return
                    self.keys.hold(key, cmd.get("vk"), self.scan_check.isChecked())
                    delay = self.press_duration.value()
                    if delay:
                        generation = self.keys._hold_generation[key]
                        QTimer.singleShot(delay, lambda k=key, g=generation: self._release_short_press(k, g))
                    else: self.keys.release(key)
                else: self.keys.hold(key, cmd.get("vk"), self.scan_check.isChecked())
                self.last_command, self.last_time = normalize(phrase), now
            except Exception as exc: self._tag_outcome(dg.OUT_SEND_ERROR); self.set_status(f"Tuş gönderilemedi: {exc}"); return
        self.last_key_sent_at = time.perf_counter()
        self._tag_outcome(dg.OUT_KEY_SENT)
        self.scope.trigger(); self.hero_key.set_text(key); self.hero_key.flash()
        self.set_status(f"‘{phrase}’ çalıştırıldı: {key} · {latency:.0f} ms")

    def _release_short_press(self, key, generation):
        try:
            if self.keys._hold_generation.get(key) == generation: self.keys.release(key)
        except Exception as exc: self.set_status(f"Tuş bırakma hatası: {exc}")

    def _on_observed(self, method, text, score, latency, stage):
        if stage == "akış":
            self.hero_latency.setToolTip("Akış bloğunun işlem süresi. Konuşma sonundan tuşa kadar uçtan uca gecikme değildir.")
        if latency > 0: self.metrics = (self.metrics + [latency])[-100:]
        if not self.is_test or self.worker_kind == "wizard": return
        target, det = normalize(self.expected_entry.text()), normalize(text) if text else ""
        if target == "-": verdict,col = ("KOMUT DIŞI KONUŞMADA TETİKLEME", "#e5484d") if det else ("KOMUT DIŞI KONUŞMA REDDEDİLDİ", "#6fcf97")
        elif stage in ("reddedildi", "kararsız", "uzak"): verdict, col = "KARARSIZ — TUŞ TETİKLENMEDİ", "#e5484d"
        elif not det: verdict, col = "EŞLEŞME YOK", DIM
        elif det == target: verdict, col = "HEDEFLE EŞLEŞTİ", "#6fcf97"
        else: verdict, col = "FARKLI İFADE", AMBER
        counts = self.validation_counts.setdefault(target, [0, 0, 0])
        counts[0 if det == target and stage not in ("kararsız", "uzak", "reddedildi") else 1 if not det else 2] += 1
        q = ("Ses uzaklığı: " + (f"{score:.2f}" if score != float("inf") else "—")) if method.startswith("örüntü") else f"Güven: {score:.2f}"
        card = QFrame(); card.setObjectName("row"); card.setStyleSheet(f"QFrame#row{{border-left:4px solid {col}}}")
        cl = QVBoxLayout(card); cl.setContentsMargins(14, 10, 14, 10); cl.setSpacing(2)
        v = label(verdict, "rowt"); v.setStyleSheet(f"color:{col};font-size:15px"); cl.addWidget(v)
        cl.addWidget(label(f"Doğru: {counts[0]} · Reddedilen: {counts[1]} · Yanlış: {counts[2]}\nYÖNTEM {method} · HEDEF {target or '—'} · DUYULAN {text or '—'}\n{q} · GECİKME " +
                           (f"{latency:.0f} ms" if latency >= 0 else "ölçülemedi") + f" · Karar: {stage}", "mono", True))
        self.results_lay.insertWidget(0, card); self.scope.trigger()

    def clear_results(self):
        self.log.clear(); self._pending_record=None
        self.validation_counts.clear()
        while self.results_lay.count() > 1:
            w = self.results_lay.takeAt(0).widget()
            if w: w.deleteLater()

    def _sens_changed(self, v):
        self.sens_label.setText("Daha seçici karar" if v > 70 else "Daha hızlı karar" if v < 35 else "Dengeli ayar")

    # ---------- system ----------
    def begin_window_pick(self):
        self.window_pick_deadline = time.monotonic() + 8.0
        self.set_status("Hedef pencereye geçin veya tıklayın. Pencere başlığı otomatik seçilip profile kaydedilecek.")
        self.window_pick_timer.start()

    def _poll_window_pick(self):
        title = self._foreground_window_title()
        if title and "autopress" not in title.casefold():
            self.window_pick_timer.stop()
            self.target_entry.setText(title)
            if self.save_profile(silent=True):
                self.set_status(f"Pencere seçildi ve profile kaydedildi: {title}")
            return
        if time.monotonic() >= self.window_pick_deadline:
            self.window_pick_timer.stop()
            self.set_status("Pencere seçimi zaman aşımına uğradı. Yeniden denemek için PENCEREYİ SEÇ düğmesine basın.")

    @staticmethod
    def _foreground_window_title():
        if sys.platform != "win32": return ""
        try:
            u = ctypes.windll.user32; hwnd = u.GetForegroundWindow(); n = u.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1); u.GetWindowTextW(hwnd, buf, n + 1)
            return buf.value.strip()
        except Exception: return ""

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self.style().standardIcon(QStyle.SP_ComputerIcon), self)
        self.tray.setToolTip("AutoPress — çift tıkla pencereyi göster")
        menu = QMenu(self)
        menu.addAction("AutoPress’i göster", self._restore_from_tray)
        menu.addSeparator()
        menu.addAction("Dinlemeyi durdur", self.stop_listening)
        menu.addAction("Tüm tuşları bırak", self.release_all)
        menu.addSeparator()
        menu.addAction("Uygulamadan çık", self.quit_app)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable(): self.tray.show()

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick: self._restore_from_tray()

    def _restore_from_tray(self):
        self.showNormal(); self.raise_(); self.activateWindow()

    def _minimize_to_tray(self):
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.hide()
            self.tray.showMessage("AutoPress", "Uygulama sistem tepsisinde çalışmaya devam ediyor.",
                                  QSystemTrayIcon.Information, 1800)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange and self.isMinimized() and hasattr(self, "tray"):
            QTimer.singleShot(0, self._minimize_to_tray)

    def quit_app(self):
        if self._running(): self.worker.stop(); self.worker.wait(2000)
        self.release_all()
        if hasattr(self, "tray"): self.tray.hide()
        QApplication.quit()

    def _game_focused(self):
        exp = self._profile().get("target_window", "").strip()
        if not exp: return True
        if sys.platform != "win32": return False
        try:
            return window_title_matches(exp, self._foreground_window_title())
        except Exception: return False

    def _guard_focus(self):
        if self.keys.held and self.focus_check.isChecked() and not self._game_focused(): self.release_all()

    def release_all(self):
        try: self.keys.release_all()
        except Exception as exc: self.set_status(f"Tuşları bırakma hatası: {exc}")

    def _update_resources(self):
        try:
            import psutil
            pr = psutil.Process(); avg = sum(self.metrics) / len(self.metrics) if self.metrics else 0
            self.resource_label.setText(f"CPU {pr.cpu_percent():.1f}%  ·  RAM {pr.memory_info().rss / 1048576:.0f} MB  ·  ort. {avg:.0f} ms ({len(self.metrics)} ölçüm)")
        except Exception: self.resource_label.setText("Kaynak ölçümü kullanılamıyor")

    def download_model(self):
        if MODEL.is_dir(): self.info("Model hazır", "Türkçe tanıma modeli zaten kurulu."); return
        self.dl_btn.setEnabled(False); self.set_status("Türkçe model indiriliyor. Bu indirme dışında tanıma internet kullanmaz.")
        def job():
            try:
                APP.mkdir(parents=True, exist_ok=True); arc = APP / "model.zip"
                urllib.request.urlretrieve(MODEL_URL, arc)
                with zipfile.ZipFile(arc) as zf: zf.extractall(APP)
                arc.unlink(missing_ok=True); self.download_done.emit(True, "Türkçe modeli indirildi.")
            except Exception as exc: self.download_done.emit(False, str(exc))
        threading.Thread(target=job, daemon=True).start()

    def _on_download(self, ok, msg):
        self.dl_btn.setEnabled(True); self.set_status(msg)
        if not ok: QMessageBox.critical(self, "İndirme başarısız", msg)

    def closeEvent(self, e):
        if QSystemTrayIcon.isSystemTrayAvailable():
            e.ignore(); self.hide()
            self.tray.showMessage("AutoPress", "Sistem tepsisinde çalışıyor. Açmak için simgeye çift tıklayın.",
                                  QSystemTrayIcon.Information, 1800)
            return
        if self._running(): self.worker.stop(); self.worker.wait(2000)
        self.release_all(); self.tray.hide(); super().closeEvent(e)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
