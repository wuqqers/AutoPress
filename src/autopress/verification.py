"""Independent spectral verification of neural decisions for similar sounds."""
from .acoustic import ShortCommandMatcher, mfcc_features


def verification_templates(neural, legacy):
    result = {}
    for phrase, template in neural.items():
        paired = template.get('phonetic_samples')
        if paired:
            result[phrase] = {'version':2,'samples':paired,'threshold':.5}
        elif legacy.get(phrase,{}).get('version') == 2:
            result[phrase] = legacy[phrase]
    # A missing rival could make a lone candidate appear artificially certain.
    return result if set(result) == set(neural) and result else {}


class PhoneticVerifier:
    def __init__(self, templates, mode='balanced'):
        self.matcher = ShortCommandMatcher(templates)
        self.last_candidate = None
        self.last_distance = float('inf')
        self.mode = mode

    def approve(self, pcm, candidate):
        try:
            self.last_candidate,self.last_distance = self.matcher.match(mfcc_features(pcm,version=2))
        except ValueError:
            self.last_candidate = None
            return False
        if self.last_candidate != candidate or not candidate:
            return False
        scores = self.matcher.last_scores
        margin = {'selective':.2,'balanced':.12,'aggressive':.06}[self.mode]
        return len(scores)==1 or scores[1][0]-scores[0][0] >= margin
