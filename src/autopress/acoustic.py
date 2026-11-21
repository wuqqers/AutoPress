"""Memory-only speech features and local DTW templates (no audio persistence)."""
from __future__ import annotations

import numpy as np
from difflib import SequenceMatcher
from functools import lru_cache

SAMPLE_RATE = 16000
FRAME = 400       # 25 ms
HOP = 160         # 10 ms
FFT_SIZE = 512
MEL_BANDS = 26
COEFFICIENTS = 13


class SpeechFeatureError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


@lru_cache(maxsize=1)
def _mfcc_constants():
    mel_points = np.linspace(0, 2595.0 * np.log10(1.0 + SAMPLE_RATE / 1400.0), MEL_BANDS + 2)
    bins = np.floor((FFT_SIZE + 1) * 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0) / SAMPLE_RATE).astype(int)
    filters = np.zeros((MEL_BANDS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for band in range(MEL_BANDS):
        left, center, right = bins[band:band + 3]
        for k in range(left, center):
            if 0 <= k < filters.shape[1]:
                filters[band, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < filters.shape[1]:
                filters[band, k] = (right - k) / (right - center)
    basis = np.cos(np.pi / MEL_BANDS * (np.arange(MEL_BANDS) + 0.5)[None, :] * np.arange(COEFFICIENTS)[:, None])
    return np.hamming(FRAME).astype(np.float32), filters, basis


def mfcc_features(pcm16: bytes, sample_rate: int = SAMPLE_RATE, *, version: int = 1) -> list[list[float]]:
    """Return normalized MFCC frames. The PCM input is consumed in memory only."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError("MFCC girişinin 16 kHz olması gerekir.")
    samples = np.frombuffer(pcm16[:len(pcm16) // 2 * 2], dtype="<i2").astype(np.float32)
    if samples.size < (960 if version == 2 else FRAME * 4):
        raise SpeechFeatureError("Örüntü için konuşma örneği çok kısa.", "too_short")
    samples /= 32768.0
    if version == 2 and np.mean(np.abs(samples) >= .995) > .01:
        raise SpeechFeatureError("Mikrofon sesi kırpılıyor; giriş sesini azaltıp yeniden söyleyin.", "clipped")
    # Remove leading/trailing low-energy frames before feature extraction.
    energy_frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
    energies = np.sqrt(np.mean(energy_frames ** 2, axis=1))
    active = np.flatnonzero(energies > max(0.003, float(energies.max()) * 0.08))
    if active.size < 4:
        raise SpeechFeatureError("Konuşma düzeyi örüntü çıkarmak için çok düşük.", "too_quiet")
    start = int(active[0] * HOP)
    end = min(samples.size, int(active[-1] * HOP) + FRAME)
    samples = samples[start:end]
    if samples.size < (960 if version == 2 else FRAME * 4):
        raise SpeechFeatureError("Konuşma örneği çok kısa.", "too_short")

    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - 0.97 * samples[:-1]
    count = 1 + max(0, (emphasized.size - FRAME) // HOP)
    indices = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    window, filters, basis = _mfcc_constants()
    frames = emphasized[indices] * window[None, :]
    power = np.abs(np.fft.rfft(frames, n=FFT_SIZE, axis=1)) ** 2

    log_mel = np.log(np.maximum(power @ filters.T, 1e-10))
    features = log_mel @ basis.T
    features = features[:, 1:]  # Drop overall energy; retain shape and spectral envelope.
    if version == 2:
        delta = np.diff(features, axis=0, prepend=features[:1])
        return np.concatenate((features, delta), axis=1).astype(np.float32).tolist()
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = np.clip((features - mean) / np.maximum(std, 1e-3), -4.0, 4.0)
    return features.astype(np.float32).tolist()


def dtw_distance(left: list[list[float]], right: list[list[float]]) -> float:
    """Length-normalized, band-limited DTW distance between two MFCC sequences."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError("Ses örüntüsü biçimi geçersiz.")
    costs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = costs.shape
    width = max(abs(n - m), int(max(n, m) * 0.2), 8)
    inf = float("inf")
    previous = np.full(m + 1, inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, inf, dtype=np.float64)
        lo, hi = max(1, i - width), min(m, i + width)
        # Unroll the horizontal recurrence into prefix sums and prefix minima.
        # This retains every DTW path in the original band without Python's
        # per-cell loop. Float64 accumulation matches the original DP precision.
        row = costs[i - 1, lo - 1:hi].astype(np.float64)
        sums = np.cumsum(row)
        before = np.empty_like(sums)
        before[0] = 0.0
        before[1:] = sums[:-1]
        incoming = np.minimum(previous[lo:hi + 1], previous[lo - 1:hi])
        current[lo:hi + 1] = sums + np.minimum.accumulate(incoming - before)
        previous = current
    return float(previous[m] / (n + m))


def build_template(samples: list[list[list[float]]]) -> dict:
    if len(samples) < 3:
        raise ValueError("Bir komut için en az 3 ses örneği gerekli.")
    nearest = []
    for i, sample in enumerate(samples):
        nearest.append(min(dtw_distance(sample, other) for j, other in enumerate(samples) if i != j))
    threshold = max(0.25, float(np.quantile(nearest, 0.9)) * 1.5 + 0.15)
    return {"version": 1, "samples": samples, "threshold": threshold}


def match_template(features: list[list[float]], templates: dict[str, dict], sensitivity: float) -> tuple[str | None, float]:
    scores = []
    for phrase, template in templates.items():
        try:
            distance = min(dtw_distance(features, sample) for sample in template["samples"])
        except (KeyError, TypeError, ValueError):
            continue
        adjusted_threshold = float(template.get("threshold", 0.5)) * (1.45 - 0.9 * sensitivity)
        scores.append((phrase, distance, adjusted_threshold))
    return _select_match(scores)


class PreparedTemplateMatcher:
    """Prepare a session snapshot once and evaluate DTW samples in bounded batches."""

    def __init__(self, templates: dict[str, dict], sensitivity: float, mode="balanced"):
        self.mode = mode
        self.version = 1
        self.normalization = None
        self.dimensions = next((len(t["samples"][0][0]) for t in templates.values()), 12)
        self.last_reason = ""
        self.entries = []
        samples = []
        for phrase, template in templates.items():
            try:
                arrays = [np.asarray(s, dtype=np.float32) for s in template['samples']]
                if not arrays or any(a.ndim != 2 or not len(a) or a.shape[1] != self.dimensions or
                                     not np.isfinite(a).all() for a in arrays):
                    continue
                threshold = float(template.get('threshold', 0.5)) * (1.45 - 0.9 * sensitivity)
            except (KeyError, TypeError, ValueError):
                continue
            owner = len(self.entries)
            self.entries.append((phrase, threshold))
            samples.extend((owner, a) for a in arrays)
        # Similar lengths reduce padding; limit temporary memory even for large profiles.
        samples.sort(key=lambda item: len(item[1]))
        self.batches = []
        for start in range(0, len(samples), 16):
            group = samples[start:start + 16]
            lengths = np.array([len(a) for _, a in group])
            padded = np.zeros((len(group), int(lengths.max()), self.dimensions), dtype=np.float32)
            for index, (_, a) in enumerate(group):
                padded[index, :len(a)] = a
            self.batches.append((np.array([owner for owner, _ in group]), lengths, padded))

    def distances(self, features):
        a = np.asarray(features, dtype=np.float32)
        if a.ndim != 2 or not len(a) or a.shape[1] != self.dimensions or not np.isfinite(a).all():
            return np.full(len(self.entries), np.inf)
        best = np.full(len(self.entries), np.inf)
        n = len(a)
        for owners, lengths, padded in self.batches:
            size, m, _ = padded.shape
            width = np.maximum.reduce([np.abs(n - lengths),
                                       (np.maximum(n, lengths) * 0.2).astype(int),
                                       np.full(size, 8)])
            columns = np.arange(1, m + 1)[None, :]
            previous = np.full((size, m + 1), np.inf)
            previous[:, 0] = 0.0
            for i, frame in enumerate(a, 1):
                valid = (columns <= lengths[:, None]) & (np.abs(columns - i) <= width[:, None])
                costs = np.linalg.norm(padded - frame, axis=2)
                sums = np.cumsum(costs, axis=1, dtype=np.float64)
                before = np.empty_like(sums)
                before[:, 0] = 0.0
                before[:, 1:] = sums[:, :-1]
                incoming = np.minimum(previous[:, 1:], previous[:, :-1]) - before
                incoming[~valid] = np.inf
                current = sums + np.minimum.accumulate(incoming, axis=1)
                current[~valid] = np.inf
                previous[:, 0] = np.inf
                previous[:, 1:] = current
            distances = previous[np.arange(size), lengths] / (n + lengths)
            np.minimum.at(best, owners, distances)
        return best

    def match(self, features):
        best = self.distances(features)
        return _select_match([(phrase, float(best[i]), threshold)
                              for i, (phrase, threshold) in enumerate(self.entries)])


def _select_match(scores):
    if not scores: return None, float("inf")
    scores.sort(key=lambda item: item[1])
    phrase, distance, threshold = scores[0]
    if distance > threshold: return None, distance
    if len(scores) > 1:
        second = scores[1]
        ratio = SequenceMatcher(None, phrase, second[0]).ratio()
        confusable = phrase[:1] == second[0][:1] and ratio >= 0.4
        separation = max(0.30, distance * 0.30) if confusable else max(0.12, distance * 0.12)
        if confusable:
            threshold *= 0.75
        if scores[1][1] - distance < separation: return None, distance
        if distance > threshold: return None, distance
    return phrase, distance


class ShortCommandMatcher:
    """Profile-wide normalization and two-example consensus for v2 features."""
    def __init__(self, templates, mode='balanced'):
        self.mode = mode
        self.last_reason = ''
        self.last_scores = []
        self.samples = {key: [np.asarray(s, dtype=np.float32) for s in t['samples']]
                        for key, t in templates.items()}
        all_frames = np.concatenate([s for group in self.samples.values() for s in group])
        self.mean = all_frames.mean(axis=0)
        self.scale = np.maximum(all_frames.std(axis=0), .5)
        # Learn which coefficients separate this user's commands. Estimate from
        # utterance means, with shrinkage and bounded weights for five-shot data.
        centers = [np.stack([s.mean(axis=0) for s in group]) for group in self.samples.values()]
        if len(centers) > 1:
            class_means = np.stack([c.mean(axis=0) for c in centers])
            between = class_means.var(axis=0)
            within = np.mean([c.var(axis=0) for c in centers], axis=0)
            ratio = between / (within + self.scale ** 2 * .1)
            weights = np.sqrt(np.clip(ratio / max(float(np.mean(ratio)), .01), .5, 2.))
            self.scale /= weights
        self.samples = {k: [self.normalize(s) for s in group] for k, group in self.samples.items()}
        indexed = {}
        self.owners = []
        for key, group in self.samples.items():
            for sample in group:
                indexed[str(len(indexed))] = {'samples': [sample], 'threshold': 1.}
                self.owners.append(key)
        self.prepared = PreparedTemplateMatcher(indexed, .65)
        self.thresholds = {}
        self.margins = {}
        self.warnings = []
        for key, group in self.samples.items():
            genuine, impostor, gaps = [], [], []
            for i, sample in enumerate(group):
                own = sorted(dtw_distance(sample, s) for j, s in enumerate(group) if i != j)
                score = float(np.mean(own[:2]))
                genuine.append(score)
                others = [self.score(sample, other) for k, other in self.samples.items() if k != key]
                if others:
                    impostor.append(min(others)); gaps.append(min(others) - score)
            threshold = max(.25, float(np.quantile(genuine, .9)) * 1.25 + .05)
            if impostor:
                # Competition is handled by the score margin below. Capping
                # acceptance at the nearest rival rejects genuine examples
                # even when the correct command clearly wins.
                if min(gaps) < .12:
                    self.warnings.append(key)
            self.thresholds[key] = threshold
            self.margins[key] = max(.08, float(np.quantile(gaps, .1)) * .25) if gaps else .08

    def normalize(self, sample):
        return np.clip((sample - self.mean) / self.scale, -6, 6).astype(np.float32)

    @staticmethod
    def score(sample, group):
        return float(np.mean(sorted(dtw_distance(sample, other) for other in group)[:2]))

    def match(self, features):
        sample = self.normalize(np.asarray(features, dtype=np.float32))
        distances = self.prepared.distances(sample)
        grouped = {key: [] for key in self.samples}
        for owner, distance in zip(self.owners, distances):
            grouped[owner].append(distance)
        scores = sorted((float(np.mean(sorted(values)[:2])), key) for key, values in grouped.items())
        self.last_scores = scores[:2]
        distance, key = scores[0]
        factor, margin_factor = {'selective': (.85, 1.5), 'balanced': (1., 1.), 'aggressive': (1.15, .5)}[self.mode]
        if distance > self.thresholds[key] * factor:
            self.last_reason = 'uzak'; return None, distance
        if len(scores) > 1 and scores[1][0] - distance < self.margins[key] * margin_factor:
            self.last_reason = 'kararsız'; return None, distance
        self.last_reason = 'kabul'
        return key, distance


def build_short_template(samples):
    if len(samples) < 5:
        raise ValueError('Beş geçerli örnek gerekli.')
    return {'version': 2, 'samples': samples, 'threshold': .5}


def find_outlier_sample(samples, *, duration_ratio=1.6, distance_ratio=2.5):
    """Eğitim kaydında diğerlerine benzemeyen örneğin indeksini döndürür (yoksa None).

    Ölçüt (sezgisel, ölçülmüş bir doğruluk iddiası değildir):
      * süre, medyan sürenin >1.6 katı veya <1/1.6'i;
      * diğer örneklere en yakın iki DTW uzaklığının ortalaması, medyanın >2.5 katı.
    Bu yalnızca yanlış kelime / kesilmiş / çift söylenmiş kayıtları yakalamak içindir.
    """
    if len(samples) < 4:
        return None
    lengths = np.array([len(x) for x in samples], dtype=float)
    median_len = float(np.median(lengths))
    ratios = np.maximum(lengths / median_len, median_len / lengths)
    scores = []
    for i, a in enumerate(samples):
        d = sorted(dtw_distance(a, b) for j, b in enumerate(samples) if i != j)
        scores.append(float(np.mean(d[:2])))
    scores = np.array(scores)
    median_score = float(np.median(scores))
    bad = [(max(ratios[i] / duration_ratio, scores[i] / max(median_score * distance_ratio, 1e-6)), i)
           for i in range(len(samples))
           if ratios[i] > duration_ratio or scores[i] > median_score * distance_ratio]
    return max(bad)[1] if bad else None


