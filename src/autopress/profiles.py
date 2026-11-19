"""Profile schema and persistence."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_PROFILE = {
    "name": "ROB RUN",
    "commands": {
        "flash": {"key": "5", "mode": "press"},
        "armor": {"key": "3", "mode": "press"},
        "zırh": {"key": "3", "mode": "press"},
        "can": {"key": "4", "mode": "press"},
        "zıpla": {"key": "SPACE", "mode": "press"},
        "eğil": {"key": "CTRL", "mode": "press"},
    },
}
VALID_KEYS = {*(str(n) for n in range(10)), "SPACE", "CTRL", "SHIFT", "ALT",
              "TAB", "ENTER", "ESC", "UP", "DOWN", "LEFT", "RIGHT",
              *(f"F{n}" for n in range(1, 13)), *(f"NUMPAD{n}" for n in range(10)),
              "NUMPAD_MULTIPLY", "NUMPAD_ADD", "NUMPAD_SUBTRACT", "NUMPAD_DECIMAL", "NUMPAD_DIVIDE",
              "MINUS", "EQUALS", "BRACKET_LEFT", "BRACKET_RIGHT", "BACKSLASH", "SEMICOLON",
              "APOSTROPHE", "GRAVE", "COMMA", "PERIOD", "SLASH",
              *(chr(n) for n in range(ord("A"), ord("Z") + 1))}

def normalize(text: str) -> str:
    # Keep Turkish dotless ı intact so grammar tokens retain their pronunciation.
    return " ".join(text.lower().replace("\u0307", "").split())

def validate_profiles(profiles: Any) -> dict[str, dict]:
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("En az bir oyun profili gerekli.")
    out = {}
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(profile, dict):
            raise ValueError("Profil adı veya içeriği geçersiz.")
        commands = profile.get("commands")
        if not isinstance(commands, dict):
            raise ValueError(f"{name}: komut listesi geçersiz.")
        checked = {}
        legacy_aliases = {}
        for phrase, action in commands.items():
            if not isinstance(phrase, str) or not phrase.strip() or not isinstance(action, dict):
                raise ValueError("Komut sözcüğü geçersiz.")
            key = str(action.get("key", "")).upper()
            mode = action.get("mode", "press")
            keycode = action.get("vk")
            if keycode is not None:
                if isinstance(keycode, bool) or not isinstance(keycode, int) or not 1 <= keycode <= 255:
                    raise ValueError("Windows tuş kodu 1–255 arasında bir tam sayı olmalı.")
            is_vk_label = key.startswith("VK_") and key[3:].isdigit() and 1 <= int(key[3:]) <= 255
            if key not in VALID_KEYS and not is_vk_label:
                raise ValueError(f"Desteklenmeyen tuş: {key}")
            if mode not in ("press", "hold"):
                raise ValueError(f"Desteklenmeyen mod: {mode}")
            norm = normalize(phrase)
            # Older releases folded Turkish ı into i; restore the built-in aliases.
            if norm == "zirh" and key == "3": norm = "zırh"
            if norm == "zipla" and key == "SPACE": norm = "zıpla"
            legacy_aliases[normalize(phrase)] = norm
            checked[norm] = {"key": key, "mode": mode, **({"vk": keycode} if keycode is not None else {})}
        patterns = profile.get("audio_templates", {})
        if not isinstance(patterns, dict):
            raise ValueError(f"{name}: kişisel ses örüntüleri geçersiz.")
        checked_patterns = {}
        for phrase, template in patterns.items():
            norm = normalize(phrase) if isinstance(phrase, str) else ""
            norm = legacy_aliases.get(norm, norm)
            if norm not in checked or not isinstance(template, dict) or template.get("version") != 1:
                raise ValueError(f"{name}: ses örüntüsü komutla eşleşmiyor.")
            samples = template.get("samples")
            if not isinstance(samples, list) or not 3 <= len(samples) <= 5:
                raise ValueError(f"{name}/{norm}: 3–5 kişisel örnek gerekli.")
            clean_samples = []
            for sample in samples:
                if not isinstance(sample, list) or not 4 <= len(sample) <= 500:
                    raise ValueError(f"{name}/{norm}: ses özellikleri geçersiz.")
                clean_frames = []
                for frame in sample:
                    if not isinstance(frame, list) or len(frame) != 12:
                        raise ValueError(f"{name}/{norm}: ses özellikleri geçersiz.")
                    values = [float(value) for value in frame]
                    if any(not math.isfinite(value) or abs(value) > 10 for value in values):
                        raise ValueError(f"{name}/{norm}: sonlu olmayan ses özelliği.")
                    clean_frames.append(values)
                clean_samples.append(clean_frames)
            threshold = float(template.get("threshold", 0.0))
            if not math.isfinite(threshold) or not 0.05 <= threshold <= 20:
                raise ValueError(f"{name}/{norm}: eşik geçersiz.")
            checked_patterns[norm] = {"version": 1, "samples": clean_samples, "threshold": threshold}
        silence_ms = profile.get("silence_ms", 150)
        if isinstance(silence_ms, bool) or not isinstance(silence_ms, int) or not 100 <= silence_ms <= 600:
            raise ValueError(f"{name}: sessizlik beklemesi 100–600 ms arasında tam sayı olmalı.")
        out[name.strip()] = {"name": name.strip(), "commands": checked,
                             "silence_ms": silence_ms,
                             "target_window": str(profile.get("target_window", "")).strip(),
                             "audio_templates": checked_patterns}
    return out

def load_profiles(path: Path) -> dict[str, dict]:
    if not path.exists():
        return validate_profiles({DEFAULT_PROFILE["name"]: DEFAULT_PROFILE})
    try:
        return validate_profiles(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Profil dosyası okunamadı: {exc}") from exc

def save_profiles(path: Path, profiles: dict[str, dict]) -> None:
    checked = validate_profiles(profiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
