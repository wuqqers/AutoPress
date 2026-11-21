"""Replay separately labelled feature captures, without microphone or key output.

Only the classifier is replayed: VAD, phonetic verification and keyboard focus
are not simulated. Use the in-app log to inspect those outcomes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .acoustic import ShortCommandMatcher
from .learning import AdaptiveMatcher, training_id
from .profiles import load_profiles, normalize


def score_records(templates, records, mode='balanced', learning=None):
    versions = {t['version'] for t in templates.values()}
    if versions == {3}:
        matcher=AdaptiveMatcher(templates,mode,learning)
        width=96
    elif versions == {2}:
        matcher=ShortCommandMatcher(templates,mode)
        width=24
    else: raise ValueError('Doğrulama sürüm 2 MFCC veya sürüm 3 sinir ağı örnekleri gerektirir.')
    out=dict.fromkeys(('n','correct_accept','wrong_key','missed','false_trigger','correct_reject','positives','negatives'),0)
    out['mode']=mode
    for record in records:
        feature=np.asarray(record.get('features'),dtype=float)
        if feature.ndim!=2 or not len(feature) or feature.shape[1]!=width or not np.isfinite(feature).all():
            raise ValueError('Doğrulama özellikleri seçilen motorla uyumsuz.')
        if not isinstance(record.get('expected'),str): raise ValueError('Her doğrulama örneğinin hedef etiketi gerekli.')
        expected=normalize(record['expected'])
        if expected and expected not in templates: raise ValueError('Doğrulama etiketi profilde yok: '+expected)
        predicted,_=matcher.match(feature)
        out['n']+=1
        if expected:
            out['positives']+=1
            out['correct_accept' if predicted==expected else 'wrong_key' if predicted else 'missed']+=1
        else:
            out['negatives']+=1
            out['false_trigger' if predicted else 'correct_reject']+=1
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles');parser.add_argument('profile');parser.add_argument('validation')
    args=parser.parse_args(argv)
    try:
        profile=load_profiles(Path(args.profiles))[args.profile]
        data=json.loads(Path(args.validation).read_text(encoding='utf-8'))
        records=data['records']
        engines={r['engine'] for r in records}
        if len(engines)!=1: raise ValueError('Dosya tek motorun doğrulama örneklerini içermeli.')
        neural=next(iter(engines))=='neural'
        templates=profile['neural_templates' if neural else 'audio_templates']
        learning=profile.get('adaptive_learning',{}) if neural else {}
        if any(r['training_id']!=training_id(templates) or r['learning_id']!=training_id(learning) for r in records):
            raise ValueError('Doğrulama sonrası eğitim/öğrenme değişmiş. Aynı profil sürümüyle yeniden ölçün.')
        print('Yalnız sınıflandırıcı tekrarı; VAD/ek ses kontrolü/tuş gönderimi yeniden çalıştırılmaz.')
        print('mod          doğru  yanlış  kaçırılan  komut dışı tetikleme')
        for mode in ('selective','balanced','aggressive'):
            r=score_records(templates,records,mode,learning)
            print(f"{mode:12} {r['correct_accept']:3}/{r['positives']:<3} {r['wrong_key']:5} {r['missed']:9} {r['false_trigger']:3}/{r['negatives']}")
    except (ValueError,KeyError,OSError,TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__=='__main__': raise SystemExit(main())
