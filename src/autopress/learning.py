"""Bounded adaptation of a personal head; original enrollment is immutable."""
import copy
import hashlib
import json
import time

import numpy as np

from .neural import MODEL_ID, NeuralMatcher

LIMITS = {'manual': 8, 'auto': 3}


def training_id(templates):
    return hashlib.sha256(json.dumps(templates, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_learning(data, templates):
    if not data: return {}
    if not isinstance(data, dict) or data.get('model_id') != MODEL_ID:
        raise ValueError('Öğrenme modeli geçersiz.')
    if data.get('training_id') != training_id(templates):
        return {}  # A new enrollment cannot inherit old pseudo-labels.
    examples = data.get('examples')
    if not isinstance(examples, list) or len(examples) > len(templates)*sum(LIMITS.values()):
        raise ValueError('Öğrenme örnek sayısı geçersiz.')
    checked, ids, counts = [], set(), {}
    for item in examples:
        if not isinstance(item, dict): raise ValueError('Öğrenme örneği geçersiz.')
        key, source, identity = item.get('label'), item.get('source'), item.get('id')
        vector = np.asarray(item.get('vector'), np.float64)
        if (key not in templates or source not in LIMITS or not isinstance(identity,str)
                or not 1 <= len(identity) <= 100 or identity in ids or vector.shape != (96,)
                or not np.isfinite(vector).all() or not .99 <= np.linalg.norm(vector) <= 1.01):
            raise ValueError('Öğrenme özelliği veya etiketi geçersiz.')
        ids.add(identity)
        counts[key,source] = counts.get((key,source),0)+1
        if counts[key,source] > LIMITS[source]: raise ValueError('Öğrenme sınırı aşıldı.')
        checked.append({'id':identity,'label':key,'source':source,'vector':vector.tolist()})
    blocked = data.get('blocked_labels', [])
    if not isinstance(blocked,list) or any(label not in templates for label in blocked):
        raise ValueError('Otomatik öğrenme duraklatma listesi geçersiz.')
    return {'model_id':MODEL_ID,'training_id':data['training_id'],'examples':checked,
            'blocked_labels':list(dict.fromkeys(blocked))}


def add_example(data, templates, event, label, source):
    if source not in LIMITS or label not in templates:
        raise ValueError('Öğrenme etiketi veya kaynağı geçersiz.')
    current = validate_learning(data, templates)
    result = copy.deepcopy(current) if current else {
        'model_id':MODEL_ID,'training_id':training_id(templates),'examples':[]}
    # A manual correction replaces the automatic label for the same utterance.
    result['examples'] = [e for e in result['examples'] if e['id'] != event['id']]
    same = [e for e in result['examples'] if e['label']==label and e['source']==source]
    if len(same) >= LIMITS.get(source,0):
        oldest = same[0]['id']
        result['examples'] = [e for e in result['examples'] if e['id'] != oldest]
    result['examples'].append({'id':event['id'],'label':label,'source':source,'vector':event['vector']})
    return validate_learning(result, templates)


class AdaptiveMatcher(NeuralMatcher):
    def __init__(self, templates, mode='balanced', learning=None):
        super().__init__(templates,mode)
        self.teacher = NeuralMatcher(templates,mode)
        self.teacher.mode = 'selective'
        examples = validate_learning(learning or {}, templates).get('examples',[])
        if examples:
            extras = self.normalize(np.array([e['vector'] for e in examples]))
            labels = np.array([self.keys.index(e['label']) for e in examples])
            weights = np.array([.5 if e['source']=='manual' else .15 for e in examples])
            # Fixed normalization and rejection radii come from initial samples.
            # Only the small classification head and similarity support adapt.
            self.x = np.concatenate((self.x,extras))
            self.labels = np.concatenate((self.labels,labels))
            weights = np.concatenate((np.ones(len(self.raw)),weights))
            xw = self.x * np.sqrt(weights[:,None])
            targets = np.eye(len(self.keys))[self.labels] * np.sqrt(weights[:,None])
            self.weights = xw.T @ np.linalg.solve(xw @ xw.T + .3*np.eye(len(xw)), targets)

    def auto_eligible(self, features, prediction):
        teacher, distance = self.teacher.match(features)
        if not prediction or teacher != prediction: return False
        scores = self.teacher.last_scores
        return distance < self.teacher.thresholds[prediction]*.45 and (
            len(scores)==1 or scores[1][0]-distance >= .25)


class AutoLearningGate:
    def __init__(self):
        self.support = {}
        self.last_added = {}

    def consider(self, event, data, now=None):
        now = time.monotonic() if now is None else now
        label = event.get('prediction')
        if not event.get('auto_eligible') or not label or label in data.get('blocked_labels',[]): return False
        group = self.support.setdefault(label,[])
        # Replaying the same embedding must not create artificial evidence.
        vector = np.asarray(event['vector'])
        stored = [e['vector'] for e in data.get('examples',[]) if e['label']==label]
        if any(np.linalg.norm(vector-np.asarray(v)) < .002 for v in group+stored): return False
        group.append(event['vector']); del group[:-3]
        if len(group) < 3 or now-self.last_added.get(label,-1e9) < 10.: return False
        self.last_added[label] = now
        group.clear()
        return True


def preserves_enrollment(templates, learning, mode='balanced'):
    baseline = NeuralMatcher(templates, mode)
    candidate = AdaptiveMatcher(templates, mode, learning)
    for key, template in templates.items():
        for sample in template['samples']:
            if baseline.match(sample)[0] == key and candidate.match(sample)[0] != key:
                return False
    return True
