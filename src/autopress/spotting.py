"""Incremental query-by-example spotting with bounded, in-memory state."""
import numpy as np

from .acoustic import FRAME, HOP, FFT_SIZE, _mfcc_constants


class StreamingFeatures:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = np.empty(0, np.float32)
        self.previous_sample = 0.
        self.previous_feature = None

    def feed(self, pcm):
        raw = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.
        if not len(raw): return []
        emphasized = raw - .97 * np.concatenate(([self.previous_sample], raw[:-1]))
        self.previous_sample = raw[-1]
        self.samples = np.concatenate((self.samples, emphasized))
        result = []
        window, filters, basis = _mfcc_constants()
        while len(self.samples) >= FRAME:
            frame = self.samples[:FRAME]
            power = np.abs(np.fft.rfft(frame * window, n=FFT_SIZE)) ** 2
            feature = (np.log(np.maximum(power @ filters.T, 1e-10)) @ basis.T)[1:]
            delta = np.zeros_like(feature) if self.previous_feature is None else feature - self.previous_feature
            result.append(np.concatenate((feature, delta)).astype(np.float32))
            self.previous_feature = feature
            self.samples = self.samples[HOP:]
        return result


class PersonalSpotter:
    """Subsequence DTW: a template may start anywhere in continuous speech.

    DP state includes the start of each path. Duration limits prevent a few
    frames stretching into a whole word. Two successive checks are required.
    """
    def __init__(self, matcher):
        self.matcher = matcher
        self.owners, arrays = [], []
        for key, samples in matcher.samples.items():
            for sample in samples:
                self.owners.append(key); arrays.append(sample)
        self.lengths = np.array([len(a) for a in arrays])
        self.padded = np.zeros((len(arrays), max(self.lengths), 24), np.float32)
        for i, sample in enumerate(arrays): self.padded[i, :len(sample)] = sample
        self.reset()

    def reset(self):
        count, width, _ = self.padded.shape
        self.dp = np.full((count, width + 1), np.inf)
        self.dp[:, 0] = 0.
        self.starts = np.zeros_like(self.dp, dtype=np.int64)
        self.frame_index = 0
        self.candidate = None
        self.recent = {}
        self.last_scores = []

    def feed(self, feature):
        self.frame_index += 1
        frame = self.matcher.normalize(feature)
        costs = np.linalg.norm(self.padded - frame, axis=2)
        sums = np.cumsum(costs, axis=1, dtype=np.float64)
        before = np.pad(sums[:, :-1], ((0, 0), (1, 0)))
        diagonal = self.dp[:, :-1] <= self.dp[:, 1:]
        incoming = np.minimum(self.dp[:, :-1], self.dp[:, 1:]) - before
        incoming_starts = np.where(diagonal, self.starts[:, :-1], self.starts[:, 1:])
        incoming_starts[:, 0] = self.frame_index
        minima = np.minimum.accumulate(incoming, axis=1)
        indices = np.maximum.accumulate(np.where(incoming == minima, np.arange(costs.shape[1]), 0), axis=1)
        self.starts[:, 1:] = np.take_along_axis(incoming_starts, indices, axis=1)
        self.dp[:, 1:] = sums + minima
        self.dp[:, 0] = 0.
        rows = np.arange(len(self.lengths))
        durations = self.frame_index - self.starts[rows, self.lengths] + 1
        scores = self.dp[rows, self.lengths] / (durations + self.lengths)
        valid = (durations >= np.maximum(4, self.lengths * .65)) & (durations <= self.lengths * 1.5)
        scores[~valid] = np.inf
        grouped = {key: [] for key in self.matcher.samples}
        for owner, score in zip(self.owners, scores): grouped[owner].append(score)
        for key, values in grouped.items():
            distance = float(np.mean(sorted(values)[:2]))
            self.recent[key] = min(self.recent.get(key, np.inf), distance)
        if self.frame_index % 6: return None
        ranked = sorted((value, key) for key, value in self.recent.items())
        self.last_scores = ranked[:2]
        self.recent = {}
        distance, key = ranked[0]
        factor, gap = {'selective': (.7, 1.8), 'balanced': (.8, 1.3), 'aggressive': (.95, 1.)}[self.matcher.mode]
        accepted = np.isfinite(distance) and distance < self.matcher.thresholds[key] * factor
        if len(ranked) > 1:
            accepted &= ranked[1][0] - distance >= self.matcher.margins[key] * gap
        candidate = key if accepted else None
        # A very clear two-template consensus may occur only at one endpoint.
        # Requiring another endpoint would systematically miss short commands.
        fire = candidate is not None and (candidate == self.candidate or
                                         distance < self.matcher.thresholds[key] * .6)
        self.candidate = candidate
        if fire:
            self.reset()
            return key, distance
        return None
