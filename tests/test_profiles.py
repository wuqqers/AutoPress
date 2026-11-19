import json
import pytest
from autopress.profiles import DEFAULT_PROFILE, validate_profiles, save_profiles, load_profiles, normalize
from autopress.keys import KeyController
from autopress.keys import INPUT, KEYBDINPUT, MOUSEINPUT
import ctypes

def test_default_mapping_and_normalization():
    profiles=validate_profiles({"ROB RUN":DEFAULT_PROFILE})
    assert profiles["ROB RUN"]["commands"]["flash"]["key"] == "5"
    assert profiles["ROB RUN"]["commands"]["zırh"]["key"] == "3"
    assert normalize("  ZIRH ") == "zirh"
    assert normalize("zırh") == "zırh"

def test_save_load_roundtrip(tmp_path):
    path=tmp_path/"profiles.json"; profiles={"ROB RUN":DEFAULT_PROFILE}
    save_profiles(path,profiles); assert load_profiles(path)==validate_profiles(profiles)


def test_silence_setting_is_persisted_per_profile(tmp_path):
    path = tmp_path / 'profiles.json'
    profiles = {'A': {'commands': {}, 'silence_ms': 125},
                'B': {'commands': {}, 'silence_ms': 300},
                'Legacy': {'commands': {}}}
    save_profiles(path, profiles)
    loaded = load_profiles(path)
    assert [loaded[name]['silence_ms'] for name in profiles] == [125, 300, 150]


@pytest.mark.parametrize('value', [99, 601, True, 150.5, '150', None])
def test_invalid_silence_setting_is_rejected(value):
    with pytest.raises(ValueError):
        validate_profiles({'A': {'commands': {}, 'silence_ms': value}})

def test_invalid_key_rejected():
    with pytest.raises(ValueError): validate_profiles({"x":{"commands":{"go":{"key":"F13","mode":"press"}}}})

def test_captured_virtual_key_code_is_saved_and_validated():
    profile = {"commands": {"skill": {"key": "F1", "vk": 112, "mode": "press"}}}
    checked = validate_profiles({"Game": profile})
    assert checked["Game"]["commands"]["skill"] == {"key": "F1", "vk": 112, "mode": "press"}
    with pytest.raises(ValueError):
        validate_profiles({"Game": {"commands": {"skill": {"key": "VK_999", "vk": 999, "mode": "press"}}}})
    assert KeyController.vk("F1") == 112
    assert KeyController.vk("VK_112") == 112

def test_windows_input_uses_the_full_platform_structure_size():
    expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    assert ctypes.sizeof(KEYBDINPUT) == (24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16)
    assert ctypes.sizeof(MOUSEINPUT) == (32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 24)
    assert ctypes.sizeof(INPUT) == expected_size

def test_hold_mode_and_calibrated_phrase_are_persistable():
    profiles=validate_profiles({"ROB RUN":{"commands":{"şimdi zıpla":{"key":"SPACE","mode":"hold"}}}})
    assert profiles["ROB RUN"]["commands"]["şimdi zıpla"] == {"key":"SPACE","mode":"hold"}

def test_audio_templates_are_profile_data_and_validated():
    sample = [[0.0] * 12 for _ in range(5)]
    profiles = validate_profiles({"ROB RUN": {"commands": {"flash": {"key": "5", "mode": "press"}},
                                   "audio_templates": {"flash": {"version": 1, "samples": [sample] * 3, "threshold": 0.5}}}})
    assert len(profiles["ROB RUN"]["audio_templates"]["flash"]["samples"]) == 3
    with pytest.raises(ValueError):
        validate_profiles({"ROB RUN": {"commands": {"flash": {"key": "5", "mode": "press"}},
                                        "audio_templates": {"other": {"version": 1, "samples": [sample] * 3, "threshold": 0.5}}}})

def test_legacy_turkish_default_aliases_keep_dotless_i():
    profile={"commands":{"zirh":{"key":"3","mode":"press"},"zipla":{"key":"SPACE","mode":"press"}}}
    normalized=validate_profiles({"ROB RUN":profile})["ROB RUN"]["commands"]
    assert "zırh" in normalized and "zıpla" in normalized
