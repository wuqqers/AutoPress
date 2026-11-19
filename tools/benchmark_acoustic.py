"""Synthetic matcher benchmark: no microphone capture or keyboard output.

Run with .venv-tk/Scripts/python.exe tools/benchmark_acoustic.py
"""
from time import perf_counter
import platform
import statistics

import numpy as np
from autopress.acoustic import PreparedTemplateMatcher, match_template


def measure(call, repeats=20):
    call()  # Warm up imports and array machinery.
    elapsed = []
    for _ in range(repeats):
        started = perf_counter()
        call()
        elapsed.append((perf_counter() - started) * 1000)
    return statistics.median(elapsed), float(np.percentile(elapsed, 95))


def main():
    print(f'Python {platform.python_version()} / NumPy {np.__version__}')
    print('6 commands x 5 samples; 12 MFCC coefficients; milliseconds (median / p95)')
    rng = np.random.default_rng(42)
    for frames in (30, 80, 150):
        features = rng.normal(size=(frames, 12)).tolist()
        templates = {str(i): {'threshold': 5, 'samples': [
            rng.normal(size=(frames + j * 3, 12)).tolist() for j in range(5)
        ]} for i in range(6)}
        matcher = PreparedTemplateMatcher(templates, .65)
        expected = match_template(features, templates, .65)
        actual = matcher.match(features)
        assert actual[0] == expected[0]
        assert abs(actual[1] - expected[1]) < 1e-10
        serial = measure(lambda: match_template(features, templates, .65))
        batched = measure(lambda: matcher.match(features))
        print(f'{frames:3} frames: serial {serial[0]:6.2f} / {serial[1]:6.2f}; '
              f'batched {batched[0]:6.2f} / {batched[1]:6.2f}')


if __name__ == '__main__':
    main()
