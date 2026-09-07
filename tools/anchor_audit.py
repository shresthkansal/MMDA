"""Anchor quality audit: phrase collisions + narration lead/lag signed error.

Answers two questions about phase0.STEP_ANCHORS without needing Colab, a GPU,
or an embedding model:

  1. Do any two steps' phrases collide (one a substring of another)? Such a
     phrase can only ever match the wrong step. Three such collisions existed
     as of 2026-09-08 and cost 100-240s of boundary error each.
  2. For every phrase actually spoken in a real transcript, how far is
     (utterance_start + lead_lag_sec) from the annotated boundary? Reports the
     signed error of the authored offset, the error if the offset were zero,
     and the offset that WOULD have been correct.

Matching is literal substring, not embedding cosine, on purpose: this audits
the phrase bank and the offsets, not the embedder's retrieval. A phrase
matching several utterances is reported as `ambiguous` -- the decoder has to
choose between them, and the spread shows how costly a wrong choice is.

IMPORTANT: a small lead/lag error is not the decoder's objective. Zeroing all
offsets improves the MAE this script reports and simultaneously makes the
decoder worse (see the comment above STEP_ANCHORS). Use this to find broken
phrases, not to tune offsets.

Usage:
    python3 tools/anchor_audit.py TRANSCRIPT ANNOTATIONS_CSV LABEL [...]

    # both takes at once
    python3 tools/anchor_audit.py \
        Take3_Transcript.txt annotations_master.csv take3 \
        Take2_Transcript.txt manual_annotations_take2.csv take2
"""
import itertools
import os
import sys

import pandas as pd

# tools/ lives inside the osce_pipeline package dir, which is itself the repo
# root -- so the import root is its PARENT (code/ locally, /content/pkg on Colab).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from osce_pipeline import phase0                      # noqa: E402
from osce_pipeline import segmentation as seg         # noqa: E402


def phrase_collisions():
    """Phrases belonging to one step that are substrings of another step's."""
    pairs = [(sa.step, ph.lower()) for sa in phase0.STEP_ANCHORS for ph in sa.phrases]
    seen, out = set(), []
    for (s1, p1), (s2, p2) in itertools.permutations(pairs, 2):
        if s1 != s2 and p1 in p2 and (p1, p2) not in seen:
            seen.add((p1, p2))
            out.append((s1, p1, s2, p2))
    return out


def audit(transcript_path, annotations_path, label):
    utterances = seg.parse_transcript(transcript_path)
    gold = {r["step_names"]: (float(r["start_time"]), float(r["end_time"]))
            for _, r in seg.performed_rows(seg.read_annotations(annotations_path)).iterrows()}

    rows = []
    for sa in phase0.STEP_ANCHORS:
        target = gold.get(sa.step)
        want_end = sa.role == "end_finding"
        for phrase in sa.phrases:
            hits = [u for u in utterances if phrase.lower() in u.text.lower()]
            if not hits:
                rows.append(dict(take=label, step=sa.step, phrase=phrase, role=sa.role,
                                 authored=sa.lead_lag_sec, n_hits=0, status="phrase_absent"))
                continue
            for u in hits:
                if target is None:
                    rows.append(dict(take=label, step=sa.step, phrase=phrase, role=sa.role,
                                     authored=sa.lead_lag_sec, n_hits=len(hits),
                                     status="step_not_performed"))
                    continue
                boundary = target[1] if want_end else target[0]
                rows.append(dict(
                    take=label, step=sa.step, phrase=phrase, role=sa.role,
                    authored=sa.lead_lag_sec, n_hits=len(hits), utt_time=u.start_sec,
                    gold=boundary,
                    measured=round(boundary - u.start_sec, 2),          # the offset that fits
                    err=round(u.start_sec + sa.lead_lag_sec - boundary, 2),  # error of the authored one
                    status="ok" if len(hits) == 1 else "ambiguous"))
    return pd.DataFrame(rows)


def main(argv):
    if len(argv) < 3 or len(argv) % 3 != 0:
        sys.exit(__doc__)

    print("=== cross-step phrase collisions ===")
    collisions = phrase_collisions()
    for s1, p1, s2, p2 in collisions:
        print(f"  {s1}'s {p1!r} is a substring of {s2}'s {p2!r}")
    print(f"  {len(collisions)} found\n")

    frames = [audit(argv[i], argv[i + 1], argv[i + 2]) for i in range(0, len(argv), 3)]
    df = pd.concat(frames, ignore_index=True)

    for label, g in df.groupby("take"):
        print(f"=== {label} ===")
        print(g["status"].value_counts().to_string())
        clean = g[g["status"] == "ok"]
        if clean.empty:
            print()
            continue
        zero_err = clean["utt_time"] - clean["gold"]
        print(f"  unambiguous phrases: {len(clean)}")
        print(f"  authored lead_lag : MAE {clean['err'].abs().mean():6.2f}s  "
              f"median {clean['err'].median():+.2f}s")
        print(f"  lead_lag = 0      : MAE {zero_err.abs().mean():6.2f}s  "
              f"median {zero_err.median():+.2f}s")
        worst = clean.reindex(clean["err"].abs().sort_values(ascending=False).index).head(10)
        print("  worst 10 by |error|:")
        print(worst[["step", "phrase", "role", "authored", "measured", "err"]].to_string(index=False))
        print()

    ambiguous = df[df["status"] == "ambiguous"]
    if not ambiguous.empty:
        print("=== phrases matching several utterances (decoder must choose) ===")
        summary = ambiguous.groupby(["take", "step", "phrase"]).agg(
            hits=("n_hits", "first"),
            best_err=("err", lambda s: round(s.abs().min(), 1)),
            worst_err=("err", lambda s: round(s.abs().max(), 1)))
        print(summary.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
