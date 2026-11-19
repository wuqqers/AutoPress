import numpy as np
import pytest

from autopress.acoustic import PreparedTemplateMatcher, build_template, dtw_distance, match_template, mfcc_features


def reference_dtw(left, right):
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    costs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = costs.shape
    width = max(abs(n - m), int(max(n, m) * 0.2), 8)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        for j in range(max(1, i - width), min(m, i + width) + 1):
            current[j] = costs[i - 1, j - 1] + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[m] / (n + m)


@pytest.mark.parametrize('n,m', [(1, 1), (5, 40), (40, 5), (30, 30), (80, 95), (150, 80), (490, 490)])
def test_fast_dtw_preserves_reference_distance(n, m):
    rng = np.random.default_rng(42)
    a = rng.normal(size=(n, 12)).tolist()
    b = rng.normal(size=(m, 12)).tolist()
    assert dtw_distance(a, b) == pytest.approx(reference_dtw(a, b), abs=1e-10, rel=1e-10)
    assert dtw_distance(a, a) == pytest.approx(0, abs=1e-10)


def synthetic_command(scale=1.0, shift=0):
    rate = 16000
    t = np.arange(rate, dtype=np.float32) / rate
    envelope = np.clip(np.sin(np.pi * np.minimum(t, 0.8) / 0.8), 0, 1)
    tone = np.sin(2 * np.pi * (180 + 40 * t) * t) + 0.35 * np.sin(2 * np.pi * 420 * t)
    samples = np.clip(scale * envelope * tone * 9000, -32768, 32767).astype("<i2")
    return np.concatenate((np.zeros(shift, dtype="<i2"), samples)).tobytes()


def test_mfcc_and_dtw_are_finite_and_time_tolerant():
    first = mfcc_features(synthetic_command())
    shifted = mfcc_features(synthetic_command(scale=0.7, shift=320))
    assert len(first) > 20 and len(first[0]) == 12
    distance = dtw_distance(first, shifted)
    assert np.isfinite(distance)
    assert distance < 0.8


def test_template_matching_rejects_far_pattern():
    samples = [mfcc_features(synthetic_command(scale=s)) for s in (0.9, 1.0, 1.1)]
    template = build_template(samples)
    match, distance = match_template(samples[0], {"flash": template}, 0.65)
    assert match == "flash"
    assert distance == pytest.approx(0.0)
    rate = 16000
    t = np.arange(rate, dtype=np.float32) / rate
    envelope = np.clip(np.sin(np.pi * np.minimum(t, 0.8) / 0.8), 0, 1)
    unrelated = (envelope * np.sin(2 * np.pi * 760 * t) * 9000).astype("<i2")
    other_features = mfcc_features(unrelated.tobytes())
    assert match_template(other_features, {"flash": template}, 0.65)[0] is None


def test_template_matching_rejects_close_confusable_commands():
    samples = [mfcc_features(synthetic_command(scale=s)) for s in (0.9, 1.0, 1.1)]
    template = build_template(samples)
    match, _ = match_template(samples[0], {"zırh": template, "zıpla": template}, 0.65)
    assert match is None


@pytest.mark.parametrize('n', [4, 35, 80, 150, 490])
def test_batched_matcher_preserves_mixed_length_scores(n):
    rng = np.random.default_rng(12)
    features = rng.normal(size=(n, 12)).tolist()
    templates = {str(i): {'threshold': 5, 'samples': [
        rng.normal(size=(m, 12)).tolist() for m in (4, 30, 75, 151, 500)
    ]} for i in range(4)}  # More than one batch, with samples spanning batches.
    actual = PreparedTemplateMatcher(templates, .65).match(features)
    expected = match_template(features, templates, .65)
    assert actual[0] == expected[0]
    assert actual[1] == pytest.approx(expected[1], abs=1e-10, rel=1e-10)


def test_prepared_matcher_accepts_rejects_and_keeps_session_snapshot():
    features = mfcc_features(synthetic_command())
    template = build_template([features] * 3)
    templates = {'flash': template}
    matcher = PreparedTemplateMatcher(templates, .65)
    assert matcher.match(features)[0] == 'flash'
    templates.clear()
    assert matcher.match(features)[0] == 'flash'
    assert PreparedTemplateMatcher({'zırh': template, 'zıpla': template}, .65).match(features)[0] is None
    assert PreparedTemplateMatcher({}, .65).match(features) == (None, float('inf'))
    assert matcher.match([]) == (None, float('inf'))
