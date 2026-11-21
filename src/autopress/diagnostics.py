"""Tanıma denemelerinin neden kabul/red edildiğini ve gerçek tepki süresini kaydeder.

Yalnızca sayısal özet ve (isteğe bağlı) MFCC özellikleri tutulur; ham ses asla tutulmaz.
Saat tanımları (karıştırılmamalı):
  speech_end_wall  : konuşmanın son sesli 10 ms bloğunun duvar-saati (perf_counter) karşılığı.
  decision_wall    : eşleştirme kararının bittiği an.
  key_sent_wall    : SendInput çağrısının döndüğü an (arayüz tarafı doldurur).
  compute_ms       : özellik çıkarma + eşleştirme CPU süresi (tepki süresi DEĞİLDİR).
  endpoint_wait_ms : bitiş sessizliği beklemesi (ayarlanabilir; tepki süresinin parçasıdır).
  decision_ms      : speech_end -> karar.      e2e_ms: speech_end -> tuş gönderimi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# --- Neden kodları -------------------------------------------------------------
NO_SPEECH = "no_speech"                    # VAD hiç tetiklenmedi (yalnızca sayaç; kayıt oluşmaz)
VAD_FRAGMENT = "vad_fragment"              # eşiği aşan ses vardı ama konuşma sayılacak kadar sürmedi
TOO_SHORT = "too_short"
TOO_QUIET = "too_quiet"
CLIPPED = "clipped"
TOO_LONG = "too_long"
REJECTED_FAR = "rejected_far"              # en yakın aday eşikten uzak
REJECTED_AMBIGUOUS = "rejected_ambiguous"  # iki aday birbirine çok yakın
ACCEPTED = "accepted"
EARLY_ACCEPTED = "early_accepted"

# --- Sonuç (tuş) kodları --------------------------------------------------------
OUT_KEY_SENT = "key_sent"
OUT_TEST_ONLY = "test_only"
OUT_BLOCKED_FOCUS = "blocked_focus"
OUT_BLOCKED_HELD = "blocked_held"
OUT_NO_MAPPING = "no_mapping"
OUT_SEND_ERROR = "send_error"
OUT_NOT_ACCEPTED = "not_accepted"
OUT_PENDING = "pending"

FEATURE_ERRORS = {TOO_SHORT, TOO_QUIET, CLIPPED, TOO_LONG, VAD_FRAGMENT}


@dataclass
class UtteranceRecord:
    uid: str
    reason: str
    top: str = ""
    distance: float = math.inf
    threshold: float = math.nan
    second: str = ""
    second_distance: float = math.inf
    margin_needed: float = math.nan
    accepted: str = ""
    peak_rms: float = 0.0
    noise_floor: float = 0.0
    duration_ms: float = 0.0
    frames: int = 0
    split_suspect: bool = False
    endpoint_wait_ms: float = math.nan
    compute_ms: float = math.nan
    speech_end_wall: Optional[float] = None
    decision_wall: Optional[float] = None
    key_sent_wall: Optional[float] = None
    outcome: str = OUT_PENDING
    engine: str = ""
    training_id: str = ""
    learning_id: str = ""
    expected: Optional[str] = None                      # deneme modunda hedef; "" = komut dışı konuşma
    features: Optional[list] = field(default=None, repr=False)

    @property
    def snr_db(self) -> float:
        if self.noise_floor <= 0 or self.peak_rms <= 0:
            return math.nan
        return 20 * math.log10(self.peak_rms / self.noise_floor)

    @property
    def decision_ms(self) -> float:
        if self.speech_end_wall is None or self.decision_wall is None:
            return math.nan
        return (self.decision_wall - self.speech_end_wall) * 1000

    @property
    def e2e_ms(self) -> float:
        if self.speech_end_wall is None or self.key_sent_wall is None:
            return math.nan
        return (self.key_sent_wall - self.speech_end_wall) * 1000

    def verdict(self) -> str:
        """Deneme modu: hedef biliniyorsa sonucu tek etikete indirger."""
        if self.expected is None:
            return "unlabelled"
        if self.reason in FEATURE_ERRORS:
            return "feature_error:" + self.reason
        decided = self.accepted
        if not self.expected:  # komut dışı konuşma
            return "false_trigger" if decided else "correct_reject"
        if decided:
            return "correct_accept" if decided == self.expected else "wrong_key"
        if self.top == self.expected:
            return "missed_right_top_" + self.reason.replace("rejected_", "")
        return "missed_wrong_top_" + self.reason.replace("rejected_", "")


def percentile(values, q: float) -> float:
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100
    lo = int(math.floor(pos)); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class UtteranceLog:
    def __init__(self, limit: int = 200):
        self.limit = limit
        self.records: list[UtteranceRecord] = []
        self.no_speech_counter = 0

    def add(self, record: UtteranceRecord) -> UtteranceRecord:
        self.records.append(record)
        if len(self.records) > self.limit:
            del self.records[0]
        return record

    def last(self) -> Optional[UtteranceRecord]:
        return self.records[-1] if self.records else None

    def clear(self):
        self.records.clear()

    def summary(self) -> dict:
        verdicts: dict[str, int] = {}
        for r in self.records:
            verdicts[r.verdict()] = verdicts.get(r.verdict(), 0) + 1
        outcomes: dict[str, int] = {}
        for r in self.records:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        reasons: dict[str, int] = {}
        for r in self.records:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
        sent = [r for r in self.records if r.key_sent_wall is not None]
        decided = [r for r in self.records if r.decision_wall is not None and r.accepted]
        return {
            "n": len(self.records),
            "reasons": reasons, "verdicts": verdicts, "outcomes": outcomes,
            "compute_ms": {"p50": percentile([r.compute_ms for r in self.records], 50),
                           "p95": percentile([r.compute_ms for r in self.records], 95)},
            "endpoint_wait_ms": {"p50": percentile([r.endpoint_wait_ms for r in self.records], 50),
                                 "p95": percentile([r.endpoint_wait_ms for r in self.records], 95)},
            "speech_end_to_decision_ms": {"p50": percentile([r.decision_ms for r in decided], 50),
                                          "p95": percentile([r.decision_ms for r in decided], 95),
                                          "n": len(decided)},
            "speech_end_to_key_ms": {"p50": percentile([r.e2e_ms for r in sent], 50),
                                     "p95": percentile([r.e2e_ms for r in sent], 95),
                                     "n": len(sent)},
        }

    def format_summary(self) -> str:
        s = self.summary()
        v = s["verdicts"]
        def g(k): return v.get(k, 0)
        miss = sum(c for k, c in v.items() if k.startswith("missed"))
        feat = sum(c for k, c in v.items() if k.startswith("feature_error"))
        def ms(d): return "—" if math.isnan(d["p50"]) else f"p50 {d['p50']:.0f} · p95 {d['p95']:.0f} ms"
        lines = [f"İzlenen söyleyiş: {s['n']} (etiketsiz canlı: {g('unlabelled')})",
                 f"Doğru kabul: {g('correct_accept')}",
                 f"Yanlış tuş seçimi: {g('wrong_key')}",
                 f"Reddedilen/kaçırılan: {miss}"
                 + (f"  (doğru aday öndeydi: {sum(c for k, c in v.items() if k.startswith('missed_right_top'))},"
                    f" yanlış aday öndeydi: {sum(c for k, c in v.items() if k.startswith('missed_wrong_top'))})" if miss else ""),
                 f"Ses özelliği çıkarılamadı (kısa/sessiz/kırpılma): {feat}",
                 f"Komut dışı konuşmada tetikleme: {g('false_trigger')} / {g('false_trigger') + g('correct_reject')}",
                 "",
                 f"Konuşma sonu → karar : {ms(s['speech_end_to_decision_ms'])} (n={s['speech_end_to_decision_ms']['n']})",
                 f"Konuşma sonu → tuş   : {ms(s['speech_end_to_key_ms'])} (n={s['speech_end_to_key_ms']['n']})",
                 f"Bitiş sessizliği beklemesi: {ms(s['endpoint_wait_ms'])}",
                 f"Hesaplama (özellik+eşleştirme): {ms(s['compute_ms'])}  ← tepki süresi değildir",
                 "", "Tuş sonuçları: " + ", ".join(f"{k}={c}" for k, c in sorted(s["outcomes"].items()))]
        return "\n".join(lines)

    def export_features(self) -> list[dict]:
        """Yalnızca özellikler ve etiketler; ham ses içermez."""
        out = []
        for r in self.records:
            if r.features is None or r.expected is None:
                continue
            out.append({"engine":r.engine,"training_id":r.training_id,"learning_id":r.learning_id,"expected": r.expected, "features": r.features, "reason": r.reason,
                        "top": r.top, "distance": r.distance if math.isfinite(r.distance) else None})
        return out
