"""Synthetic correctness tests for segmentation.viterbi_decode / path_to_segments.

Pure numpy -- hand-built emission matrices, no embeddings, no Colab, no GPU.
Covers the three challenge patterns that are each independently confirmed
present in Take 3's own annotations_master.csv, not hypothetical:

  A. Concurrent      -- a CO_LOCATED_GROUPS meta-state must expand to one
                        Segment per member, all sharing identical frames.
  B. Out-of-order    -- a legitimately resumed earlier step (step_14, flagged
                        "Done out of checklist order") must stay admissible
                        under the soft backward penalty, while flat emission
                        must NOT induce spurious backward moves.
  C. Fragmented      -- a step whose evidence appears in two disjoint windows
                        (step_28_2_Time_Carotid_Pulse, two rows in the CSV)
                        must yield two Segments, not one merged block.
  D. Duration bounds -- d_min/d_max must actually bind, and genuinely
                        unreachable bounds must raise rather than fail silent.

Run from the repo's parent `code/` dir:
    PYTHONPATH=. python3 osce_pipeline/tests/test_segmentation_synthetic.py

Exits non-zero if any check fails. Not pytest -- osce_pipeline has no test
framework or requirements.txt (see CLAUDE.md); this is a standalone script by
the same convention as the rest of the package.
"""
import sys
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from osce_pipeline import segmentation as seg

FAIL = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  | " + extra) if extra else ""))
    if not cond: FAIL.append(name)

def mk_meta(spec):
    """spec: list of member-lists -> synthetic MetaStates"""
    return [seg.MetaState(id=i, members=m) for i, m in enumerate(spec)]

def emission_from_windows(T, M, windows, hi=2.0, lo=0.0):
    """windows: list of (state, t0, t1) inclusive -> raw log-emission"""
    e = np.full((T, M), lo)
    for c, t0, t1 in windows:
        e[t0:t1+1, c] = hi
    return e

WIDE = None  # set per-case

# ============================================================
print("\n=== CASE A: concurrent (co-located group expansion) ===")
ms = seg.build_meta_states()
M = len(ms)
T = 470  # 10 frames per meta-state
win = [(c, c*10, c*10+9) for c in range(M)]
loge = emission_from_windows(T, M, win, hi=5.0, lo=0.0)
bounds = [(0.001, 0.9)] * M
path = seg.viterbi_decode(loge, bounds)
segs = seg.path_to_segments(path, ms, fps=24.0)
g0 = [s for s in segs if s.step in ms[0].members]
check("greeting cluster expands to 6 Segments", len(g0) == 6, f"got {len(g0)}: {[s.step for s in g0]}")
frames = {(s.start_frame, s.end_frame) for s in g0}
check("all 6 share identical start/end frames", len(frames) == 1, f"windows={frames}")
check("decoded path is monotonic non-decreasing", all(path[i] <= path[i+1] for i in range(T-1)))
check("path visits all 47 meta-states", len(set(path)) == M, f"visited {len(set(path))}")

# ============================================================
print("\n=== CASE B: out-of-order resumption (step_14 pattern) ===")
spec = [["s0"], ["s1"], ["s2"], ["s3"], ["s4"]]
ms_b = mk_meta(spec); Mb = 5
# evidence order: 0,1,2,3, then RESUME 1, then 4
T = 120
win = [(0,0,19),(1,20,39),(2,40,59),(3,60,79),(1,80,99),(4,100,119)]
loge = emission_from_windows(T, Mb, win, hi=5.0, lo=0.0)
bounds = [(0.001, 0.9)] * Mb
path = seg.viterbi_decode(loge, bounds)
mid = path[80:100]
check("resumption window decodes back to state 1", set(mid) == {1}, f"got {sorted(set(mid))}")
back = [(path[i], path[i+1]) for i in range(T-1) if path[i+1] < path[i]]
check("exactly one backward transition taken", len(back) == 1, f"backward transitions={back}")

print("\n  -- flat emission control --")
flat = np.zeros((T, Mb))
path_flat = seg.viterbi_decode(flat, bounds)
back_flat = [(path_flat[i], path_flat[i+1]) for i in range(T-1) if path_flat[i+1] < path_flat[i]]
check("flat emission -> no backward transitions", len(back_flat) == 0, f"got {back_flat}")
check("flat emission -> monotonic", all(path_flat[i] <= path_flat[i+1] for i in range(T-1)))

print("\n  -- backward-penalty sensitivity --")
for br in [0.05, 0.6, 5.0, 50.0]:
    p = seg.viterbi_decode(loge, bounds, backward_rate=br)
    resumed = set(p[80:100]) == {1}
    print(f"     backward_rate={br:<5} resumption taken: {resumed}")

# ============================================================
print("\n=== CASE C: fragmented / revisited step (step_28_2 pattern) ===")
segs_b = seg.path_to_segments(path, ms_b, fps=24.0)
s1 = [s for s in segs_b if s.step == "s1"]
check("s1 yields 2 disjoint Segments", len(s1) == 2, f"got {len(s1)}: {[(x.start_frame,x.end_frame) for x in s1]}")
if len(s1) == 2:
    check("the two s1 windows are disjoint", s1[0].end_frame < s1[1].start_frame,
          f"{(s1[0].start_frame,s1[0].end_frame)} vs {(s1[1].start_frame,s1[1].end_frame)}")
    check("separated by another step's segment",
          any(x.start_frame > s1[0].end_frame and x.end_frame < s1[1].start_frame for x in segs_b))

print("\n  -- fragmented co-located group --")
ms_c = mk_meta([["a1","a2"], ["b"], ["c"]]); Mc = 3
T = 90
loge_c = emission_from_windows(T, Mc, [(0,0,29),(1,30,49),(0,50,69),(2,70,89)], hi=5.0)
path_c = seg.viterbi_decode(loge_c, [(0.001,0.9)]*Mc)
segs_c = seg.path_to_segments(path_c, ms_c, fps=24.0)
a1 = [s for s in segs_c if s.step=="a1"]; a2 = [s for s in segs_c if s.step=="a2"]
check("revisited group -> 2 Segments each for a1 and a2", len(a1)==2 and len(a2)==2, f"a1={len(a1)} a2={len(a2)}")
check("both members share frames in BOTH occurrences",
      len(a1)==2 and len(a2)==2 and
      (a1[0].start_frame,a1[0].end_frame)==(a2[0].start_frame,a2[0].end_frame) and
      (a1[1].start_frame,a1[1].end_frame)==(a2[1].start_frame,a2[1].end_frame))

# ============================================================
print("\n=== CASE D: duration constraints actually bind ===")
ms_d = mk_meta([["x0"],["x1"],["x2"]]); Md = 3
T = 90
loge_d = emission_from_windows(T, Md, [(0,0,29),(1,30,59),(2,60,89)], hi=5.0)
# d_max forces state 0 to be left early even though emission wants 30 frames
p_tight = seg.viterbi_decode(loge_d, [(0.001,0.10),(0.001,0.9),(0.001,0.9)])
run0 = sum(1 for i,c in enumerate(p_tight) if c==0 and all(p_tight[j]==0 for j in range(i+1)))
check("d_max caps state 0's run (<=9 frames of 30)", run0 <= 9, f"run0={run0} frames, d_max=0.10*90=9")
# d_min forces state 0 to be held longer than emission wants
p_min = seg.viterbi_decode(loge_d, [(0.5,0.9),(0.001,0.9),(0.001,0.9)])
run0b = sum(1 for i,c in enumerate(p_min) if c==0 and all(p_min[j]==0 for j in range(i+1)))
check("d_min holds state 0 >=45 frames (of 30 evidence)", run0b >= 45, f"run0={run0b} frames, d_min=0.5*90=45")
# NOTE: bounds like (0.9, 0.95) on every state are NOT infeasible -- forward
# transitions are soft, so the decoder legally SKIPS intermediate states to
# reach the end. Only bounds that make a state impossible to ever leave
# (d_min > 1.0) make the end state truly unreachable.
p_skip = seg.viterbi_decode(loge_d, [(0.9,0.95)]*3)
check("tight-but-satisfiable bounds decode by skipping, not by failing",
      set(p_skip) == {0, 2}, f"visited {sorted(set(p_skip))} (state 1 skipped by design)")
try:
    seg.viterbi_decode(loge_d, [(1.5,2.0),(0.001,0.9),(0.001,0.9)])
    check("truly unreachable bounds raise RuntimeError", False, "no exception raised")
except RuntimeError as e:
    check("truly unreachable bounds raise RuntimeError", True, str(e)[:55]+"...")

print("\n=== CASE E: skipping is exactly cost-neutral (known finding) ===")
fr = seg.DEFAULT_FORWARD_RATE
traverse = sum(seg.transition_penalty(1, fr, 0.6) for _ in range(4))
skip = seg.transition_penalty(4, fr, 0.6)
check("linear forward penalty makes skip == traverse (documented finding)",
      abs(traverse - skip) < 1e-12, f"traverse={traverse:.4f} skip={skip:.4f}")
ms_e = seg.build_meta_states(); Me = len(ms_e); Te = 13000
import os
OB = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Obsidian")
csv = os.path.join(OB, "annotations_master.csv")
if os.path.exists(csv):
    b_real = seg.estimate_duration_bounds(csv, ms_e, total_duration_sec=541.5)
    check("real Take 3 bound set is feasible (sum(d_min) <= 1.0)",
          sum(x[0] for x in b_real) <= 1.0, f"sum(d_min)={sum(x[0] for x in b_real):.4f}")
    vis_flat = len(set(seg.viterbi_decode(np.zeros((Te, Me)), b_real)))
    e_bump = np.zeros((Te, Me)); w = Te // Me
    for c in range(Me): e_bump[c*w:(c+1)*w, c] = 0.01
    vis_bump = len(set(seg.viterbi_decode(e_bump, b_real)))
    print(f"     flat emission -> {vis_flat}/{Me} visited; 0.01 bump -> {vis_bump}/{Me}")
    check("a tiny (0.01) emission signal is enough to visit every meta-state",
          vis_bump == Me, f"got {vis_bump}/{Me}")
else:
    print(f"     SKIPPED real-bounds checks: {csv} not found")

print("\n" + "="*55)
print("FAILURES:", FAIL if FAIL else "none - all synthetic checks passed")
sys.exit(1 if FAIL else 0)
