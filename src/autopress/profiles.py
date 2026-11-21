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
            if norm not in checked or not isinstance(template, dict) or template.get("version") not in (1, 2, 3):
                raise ValueError(f"{name}: ses örüntüsü komutla eşleşmiyor.")
            samples = template.get("samples")
            if not isinstance(samples, list) or not 3 <= len(samples) <= 5:
                raise ValueError(f"{name}/{norm}: 3–5 kişisel örnek gerekli.")
            neural = template['version'] == 3
            if neural:
                from .neural import MODEL_ID
                if template.get('model_id') != MODEL_ID: raise ValueError('Sinir ağı model sürümü uyumsuz.')
            clean_samples = []
            for sample in samples:
                if not isinstance(sample, list) or not ((len(sample) == 1) if neural else (4 <= len(sample) <= 500)):
                    raise ValueError(f"{name}/{norm}: ses özellikleri geçersiz.")
                clean_frames = []
                for frame in sample:
                    if not isinstance(frame, list) or len(frame) != (96 if neural else 24 if template["version"] == 2 else 12):
                        raise ValueError(f"{name}/{norm}: ses özellikleri geçersiz.")
                    values = [float(value) for value in frame]
                    if any(not math.isfinite(value) or abs(value) > (1000 if template["version"] == 2 else 10) for value in values):
                        raise ValueError(f"{name}/{norm}: sonlu olmayan ses özelliği.")
                    clean_frames.append(values)
                clean_samples.append(clean_frames)
            threshold = float(template.get("threshold", 0.0))
            if not math.isfinite(threshold) or not 0.05 <= threshold <= 20:
                raise ValueError(f"{name}/{norm}: eşik geçersiz.")
            checked_patterns[norm] = {**({"model_id": template["model_id"]} if neural else {}),"version": template["version"], "samples": clean_samples, "threshold": threshold}
            if neural and 'phonetic_samples' in template:
                paired = {'version':2,'samples':template['phonetic_samples'],'threshold':.5}
                checked_pair = validate_profiles({name:{'commands':commands,'audio_templates':{phrase:paired}}})[name]['audio_templates'][norm]
                checked_patterns[norm]['phonetic_samples'] = checked_pair['samples']
        pending = profile.get("pending_audio_templates", {})
        if pending:
            staged = validate_profiles({name: {"commands": commands, "audio_templates": pending}})[name]
            pending = staged["audio_templates"]
        neural_maps = {}
        for field in ('neural_templates', 'pending_neural_templates'):
            entries = profile.get(field, {})
            if not isinstance(entries, dict): raise ValueError('Sinir ağı kayıtları geçersiz.')
            if any(not isinstance(t, dict) or t.get('version') != 3 for t in entries.values()):
                raise ValueError('Sinir ağı alanında eski örnek sürümü var.')
            neural_maps[field] = validate_profiles({name: {'commands': commands, 'audio_templates': entries}})[name]['audio_templates'] if entries else {}
        continuous = profile.get("continuous_spotting", False)
        from .learning import validate_learning
        learning = validate_learning(profile.get('adaptive_learning', {}), neural_maps['neural_templates'])
        verify_sounds = profile.get('verify_similar_sounds', True)
        if not isinstance(verify_sounds, bool): raise ValueError('Ses doğrulaması ayarı geçersiz.')
        auto_learning = profile.get('auto_learning', True)
        if not isinstance(auto_learning, bool): raise ValueError('Otomatik öğrenme ayarı geçersiz.')
        if not isinstance(continuous, bool): raise ValueError("Akış algılama ayarı geçersiz.")
        mode = profile.get("decision_mode", "balanced")
        if mode not in ("selective", "balanced", "aggressive"):
            raise ValueError("Karar modu geçersiz.")
        silence_ms = profile.get("silence_ms", 150)
        if isinstance(silence_ms, bool) or not isinstance(silence_ms, int) or not 100 <= silence_ms <= 600:
            raise ValueError(f"{name}: sessizlik beklemesi 100–600 ms arasında tam sayı olmalı.")
        out[name.strip()] = {**neural_maps, "name": name.strip(), "commands": checked,
                             "adaptive_learning": learning, "auto_learning": auto_learning,
                             "verify_similar_sounds": verify_sounds,
                             "silence_ms": silence_ms, "decision_mode": mode,
                             "early_commands": [x for x in profile.get("early_commands", []) if isinstance(x, str) and x in checked],
                             "pending_audio_templates": pending, "continuous_spotting": continuous,
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
