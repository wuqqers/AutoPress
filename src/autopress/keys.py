"""Windows SendInput key controller; no keyboard hook or macro injection."""
import ctypes
from ctypes import wintypes

_MAP = {"SPACE": 0x20, "CTRL": 0x11, "SHIFT": 0x10, "ALT": 0x12,
        "TAB": 0x09, "ENTER": 0x0D, "ESC": 0x1B, "UP": 0x26,
        "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27}
_MAP.update({f"F{n}": 0x70 + n - 1 for n in range(1, 13)})
_MAP.update({f"NUMPAD{n}": 0x60 + n for n in range(10)})
_MAP.update({"NUMPAD_MULTIPLY": 0x6A, "NUMPAD_ADD": 0x6B, "NUMPAD_SUBTRACT": 0x6D,
             "NUMPAD_DECIMAL": 0x6E, "NUMPAD_DIVIDE": 0x6F, "MINUS": 0xBD,
             "EQUALS": 0xBB, "BRACKET_LEFT": 0xDB, "BRACKET_RIGHT": 0xDD,
             "BACKSLASH": 0xDC, "SEMICOLON": 0xBA, "APOSTROPHE": 0xDE,
             "GRAVE": 0xC0, "COMMA": 0xBC, "PERIOD": 0xBE, "SLASH": 0xBF})
ULONG_PTR = ctypes.c_size_t
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]
class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]
class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]
class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]

class KeyController:
    def __init__(self):
        self.held: set[str] = set()
        self._held_vks: dict[str, int] = {}
        self._held_scans: dict[str, bool] = {}
        self._hold_generation: dict[str, int] = {}
        self._next_generation = 0
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.user32.SendInput.restype = wintypes.UINT
    @staticmethod
    def vk(key: str) -> int:
        key = key.upper()
        if key in _MAP: return _MAP[key]
        if key.startswith("VK_") and key[3:].isdigit():
            value = int(key[3:])
            if 1 <= value <= 255: return value
        if len(key) == 1 and key.isalnum(): return ord(key)
        raise ValueError(f"Desteklenmeyen tuş: {key}")
    def _send(self, key: str, up: bool, keycode: int | None = None, scan_mode: bool = False) -> None:
        flags = 0x0002 if up else 0
        code = int(keycode) if keycode is not None else self.vk(key)
        if not 1 <= code <= 255: raise ValueError(f"Windows tuş kodu geçersiz: {code}")
        scan = 0
        if scan_mode:
            scan = int(self.user32.MapVirtualKeyW(code, 4))
            if not scan: raise ValueError(f"Tuş için tarama kodu bulunamadı: {key}")
            if scan & 0xFF00 in (0xE000, 0xE100): flags |= 0x0001
            scan &= 0xFF
            flags |= 0x0008
        item = INPUT(type=1, union=INPUT_UNION(ki=KEYBDINPUT(0 if scan_mode else code, scan, flags, 0, 0)))
        if self.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())
    def press(self, key: str, keycode: int | None = None) -> None:
        self._send(key, False, keycode)
        self.held.add(key)
        self._held_vks[key] = int(keycode) if keycode is not None else self.vk(key)
        self.release(key)
    def hold(self, key: str, keycode: int | None = None, scan_mode: bool = False) -> None:
        if key not in self.held:
            self._send(key, False, keycode, scan_mode)
            self.held.add(key)
            self._held_vks[key] = int(keycode) if keycode is not None else self.vk(key)
            self._held_scans[key] = scan_mode
            self._next_generation += 1
            self._hold_generation[key] = self._next_generation
    def release(self, key: str) -> None:
        if key in self.held:
            self._send(key, True, self._held_vks.get(key), self._held_scans.get(key, False))
            self.held.discard(key)
            self._held_vks.pop(key, None)
            self._held_scans.pop(key, None)
            self._hold_generation.pop(key, None)
    def release_all(self) -> None:
        errors = []
        for key in tuple(self.held):
            try: self.release(key)
            except Exception as exc: errors.append(exc)
        if errors: raise errors[0]
