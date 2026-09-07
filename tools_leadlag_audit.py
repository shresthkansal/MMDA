"""Narration lead/lag signed-error audit.

For every STEP_ANCHORS phrase, find where it is literally spoken in a real
transcript and compare (utterance_start + authored lead_lag_sec) against the
gold boundary from the manual annotations. Yields the MEASURED lead_lag that
would have been correct, per anchor.

Literal substring matching (not embeddings) on purpose: this audits the
authored lead_lag values, not the embedder's retrieval.
"""
import sys, os, re, json
from collections import defaultdict
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "Desktop", "MMDA", "code"))
sys.path.insert(0, "/Users/shresthkansal/Desktop/MMDA/code")

from osce_pipeline import phase0
from osce_pipeline import segmentation as seg


def load_gold(csv_path):
    df = seg.read_annotations(csv_path)
    df = seg.performed_rows(df)
    gold = {}
    for _, r in df.iterrows():
        gold[r["step_names"]] = (float(r["start_time"]), float(r["end_time"]))
    return gold


def audit(transcript_path, annotations_path, take_label):
    utts = seg.parse_transcript(transcript_path)
    gold = load_gold(annotations_path)
    rows = []
    for sa in phase0.STEP_ANCHORS:
        g = gold.get(sa.step)
        target_kind = "start" if sa.role != "end_finding" else "end"
        for phrase in sa.phrases:
            p = phrase.lower()
            hits = [u for u in utts if p in u.text.lower()]
            if not hits:
                rows.append(dict(take=take_label, step=sa.step, phrase=phrase, role=sa.role,
                                 authored=sa.lead_lag_sec, n_hits=0, utt_time=None,
                                 gold=None if g is None else g[0 if target_kind == "start" else 1],
                                 measured=None, err=None, status="phrase_absent"))
                continue
            for u in hits:
                if g is None:
                    rows.append(dict(take=take_label, step=sa.step, phrase=phrase, role=sa.role,
                                     authored=sa.lead_lag_sec, n_hits=len(hits), utt_time=u.start_sec,
                                     gold=None, measured=None, err=None, status="step_not_performed"))
                    continue
                target = g[0] if target_kind == "start" else g[1]
                measured = target - u.start_sec          # the lead_lag that WOULD be right
                err = (u.start_sec + sa.lead_lag_sec) - target   # signed error of authored value
                rows.append(dict(take=take_label, step=sa.step, phrase=phrase, role=sa.role,
                                 authored=sa.lead_lag_sec, n_hits=len(hits), utt_time=u.start_sec,
                                 gold=target, measured=round(measured, 2), err=round(err, 2),
                                 status="ok" if len(hits) == 1 else "ambiguous"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    specs = [
        ("/Users/shresthkansal/Desktop/MMDA/Obsidian/Take3_Transcript_Full.txt",
         "/Users/shresthkansal/Desktop/MMDA/Obsidian/annotations_master.csv", "take3"),
    ]
    extra = sys.argv[1:]
    if len(extra) == 3:
        specs.append((extra[0], extra[1], extra[2]))
    for t, a, lab in specs:
        out.append(audit(t, a, lab))
    df = pd.concat(out, ignore_index=True)
    df.to_csv("/private/tmp/claude-501/-Users-shresthkansal-Desktop-MMDA/2b3c3ae8-06b8-48ff-90b1-9ad6db8d83dc/scratchpad/leadlag_audit.csv", index=False)

    print(f"anchors total: {len(phase0.STEP_ANCHORS)}  phrase-rows: {len(df)}")
    for lab, g in df.groupby("take"):
        print(f"\n===== {lab} =====")
        print(g["status"].value_counts().to_string())
        ok = g[g["status"].isin(["ok", "ambiguous"])].copy()
        if ok.empty:
            continue
        print(f"\nsigned error of AUTHORED lead_lag (sec), n={len(ok)}:")
        print(f"  mean {ok['err'].mean():+.2f}   median {ok['err'].median():+.2f}   "
              f"MAE {ok['err'].abs().mean():.2f}   p90|err| {ok['err'].abs().quantile(0.9):.2f}")
        base = ok["utt_time"] - ok["gold"]
        print(f"raw narration offset (utterance - gold), i.e. error with lead_lag=0:")
        print(f"  mean {base.mean():+.2f}   median {base.median():+.2f}   "
              f"MAE {base.abs().mean():.2f}   p90|err| {base.abs().quantile(0.9):.2f}")
        print("\nworst 15 authored lead_lags by |err|:")
        w = ok.reindex(ok["err"].abs().sort_values(ascending=False).index).head(15)
        print(w[["step", "phrase", "role", "authored", "measured", "err", "n_hits"]].to_string(index=False))
