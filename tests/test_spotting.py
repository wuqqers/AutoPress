import numpy as np

from autopress.acoustic import ShortCommandMatcher, build_short_template
from autopress.spotting import PersonalSpotter, StreamingFeatures


def test_stream_features_ignore_callback_boundaries():
    pcm = np.random.default_rng(4).integers(-8000, 8000, 8000, dtype=np.int16).tobytes()
    full = StreamingFeatures().feed(pcm)
    stream = StreamingFeatures()
    split = []
    for start in range(0, len(pcm), 320): split.extend(stream.feed(pcm[start:start+320]))
    assert np.allclose(full, split, atol=1e-5)


def test_commands_inside_unbroken_feature_stream_and_repetition():
    rng = np.random.default_rng(5)
    q = rng.normal(size=(30, 24)).astype(np.float32)
    w = rng.normal(5, 1, size=(60, 24)).astype(np.float32)
    # Varied enrollments produce a usable acceptance radius.
    templates = {key: build_short_template([(sample + rng.normal(0, .15, sample.shape)).tolist()
                                           for _ in range(5)]) for key, sample in [('q', q), ('w', w)]}
    spotter = PersonalSpotter(ShortCommandMatcher(templates))
    stream = np.concatenate([rng.normal(-5, 1, (40,24)), q, q[-8:],
                             rng.normal(-5, 1, (40,24)), w, w[-8:],
                             rng.normal(-5, 1, (40,24)), q, q[-8:]])
    results = [result[0] for frame in stream if (result := spotter.feed(frame))]
    assert results == ['q', 'w', 'q']


def test_background_and_identical_commands_do_not_trigger():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=(30, 24)).tolist()
    template = build_short_template([sample] * 5)
    spotter = PersonalSpotter(ShortCommandMatcher({'q': template, 'w': template}))
    assert all(spotter.feed(frame) is None for frame in sample * 4)


def test_unrelated_stream_and_reset_drop_incomplete_command():
    rng = np.random.default_rng(19)
    sample = rng.normal(size=(50, 24))
    matcher = ShortCommandMatcher({'test': build_short_template([sample.tolist()] * 5)})
    spotter = PersonalSpotter(matcher)
    assert all(spotter.feed(frame) is None for frame in rng.normal(8, 1, (500,24)))
    for frame in sample[:25]: spotter.feed(frame)
    spotter.reset()
    assert all(spotter.feed(frame) is None for frame in sample[25:])
