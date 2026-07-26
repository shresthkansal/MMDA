"""
FS_model's own LLM-adjacent grading prototype (Whisper transcript matching +
geometric visual-zone check against a rubric).

FLAGGED: this functionally overlaps with Data_Preparation.ipynb's Qwen-VLM +
embeddings grading approach, which has already been established (backed by
a medRxiv multimodal-OSCE paper) as the superior method going forward. Kept
here, isolated from the rest of the pipeline, per "keep unique working code
now, judge later" -- expect this module to be dropped or kept only for
comparison once Step 2 (pruning) happens.

Ported from FS_model.ipynb cell idx 84 (the later of two near-identical
`Examiner` versions -- idx 84 is a strict superset of idx 83, adding
verification-snippet export; both ran successfully, idx 84 is preferred).

Depends on `action_master_dictionary.json` (see action_library.py) as its
rubric, and on the same known-weak `predict_zone_robust` geometric visual
check flagged in action_library.py.
"""
import json
import os
import re

import numpy as np
import pandas as pd

CHECKLIST_TEXT = """
    Cardiovascular examination context. Key terms:
    Greets patient, introduces self, builds rapport, obtains consent, hand hygiene,
    position 45 degrees, angle of Louis, JVP estimation, jugular venous pressure,
    general inspection, cyanosis, breathlessness, Turner's syndrome, Marfan's syndrome,
    Trisomy, thyrotoxicosis, acromegaly, midline scar, axillary scar, pacemaker, AICD,
    pectus excavatum, apex beat, clubbing, splinter haemorrhages, Osler's nodes,
    Janeway lesions, tendon xanthoma, nicotine staining, peripheral cyanosis,
    radial pulse, radio-radial delay, radio-femoral delay, collapsing pulse,
    water hammer pulse, Corrigan's sign, abdominojugular reflux, carotid bruit,
    parasternal heave, thrill, palpable P2, palpable A2, mitral area, tricuspid area,
    pulmonary area, aortic area, diaphragm, bell, mid-diastolic murmur, systolic murmur,
    pan-systolic murmur, ejection systolic murmur, opening snap, tumor plop,
    respiratory manoeuver, Valsalva maneuver, deep breath, hold breath, lean forward,
    sacral edema, pedal edema, saphenous vein graft, fundoscopy, hematuria.
"""
CLEAN_PROMPT = " ".join(CHECKLIST_TEXT.split())


# ==========================================
# Geometric visual-zone check (same known-weak heuristic as action_library.py)
# ==========================================
def get_point(row, part: str) -> np.ndarray:
    return np.array([row.get(f"{part}_X", 0), row.get(f"{part}_Y", 0)])


def get_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def predict_zone_robust(doc_row, pat_row, frame_width: int = 1920) -> str:
    if doc_row is None or pat_row is None:
        return "No Data"

    doc_wrist_r, doc_wrist_l = get_point(doc_row, "R_Wrist"), get_point(doc_row, "L_Wrist")
    doc_sources = [p for p in [doc_wrist_r, doc_wrist_l] if p[0] != 0]

    p_nose = get_point(pat_row, "Nose")
    p_neck = (get_point(pat_row, "R_Shoulder") + get_point(pat_row, "L_Shoulder")) / 2
    p_chest = (p_neck + (get_point(pat_row, "R_Hip") + get_point(pat_row, "L_Hip")) / 2) / 2
    p_stomach = (get_point(pat_row, "R_Hip") + get_point(pat_row, "L_Hip")) / 2

    ruler = get_dist(get_point(pat_row, "R_Shoulder"), get_point(pat_row, "L_Shoulder"))
    if ruler < 20:
        ruler = frame_width * 0.15
    scale = ruler * 0.8

    def min_d(targets) -> float:
        best = 9999.0
        valid_t = [t for t in targets if t[0] != 0]
        if not valid_t:
            return 9999.0
        for t in valid_t:
            for s in doc_sources:
                best = min(best, get_dist(s, t))
        return best

    zones = {
        "Face": (min_d([p_nose]), 0.5),
        "Chest": (min_d([p_chest, p_neck]), 0.6),
        "Stomach": (min_d([p_stomach]), 0.6),
    }

    best_zone, min_norm = "Unknown", 999.0
    for z, (raw, thresh) in zones.items():
        norm = raw / (thresh * scale)
        if norm < min_norm:
            min_norm = norm
            best_zone = z

    return f"Examining: {best_zone}" if min_norm < 0.50 else "Observing"


# ==========================================
# Grading engine
# ==========================================
class Examiner:
    """Grades a student take against the master rubric via Whisper transcript
    keyword-matching + geometric visual-zone checking. Optionally exports
    verification video snippets for a chosen subset of steps."""

    def __init__(self, master_dict_path: str, student_video_path: str,
                 student_keypoints_path: str, student_timing_path: str,
                 output_snippet_dir: str = None, snippet_targets: list = None,
                 whisper_model_size: str = "medium"):
        import moviepy.editor as mp
        import whisper

        print("\nInitializing AI Examiner...")

        self.output_snippet_dir = output_snippet_dir
        self.snippet_targets = snippet_targets or []
        if self.output_snippet_dir and not os.path.exists(self.output_snippet_dir):
            os.makedirs(self.output_snippet_dir)

        with open(master_dict_path, "r") as f:
            self.rubric = json.load(f)
        print(f"   Rubric loaded: {len(self.rubric)} steps defined.")

        self.timings = pd.read_csv(student_timing_path)
        self.keypoints = pd.read_csv(student_keypoints_path)
        self.video = mp.VideoFileClip(student_video_path)
        self.fps = self.video.fps
        print(f"   Video loaded: {self.video.duration:.1f}s @ {self.fps}fps")

        print("   Loading Whisper model...")
        self.model = whisper.load_model(whisper_model_size, device="cuda")
        print("   Ready to grade.")

    def save_video_snippet(self, start_t: float, end_t: float, filename: str) -> bool:
        path = os.path.join(self.output_snippet_dir, filename)
        if not os.path.exists(path):
            try:
                clip = self.video.subclip(start_t, end_t)
                clip.write_videofile(path, codec="libx264", audio_codec="aac",
                                      verbose=False, logger=None, preset="ultrafast")
                return True
            except Exception as e:
                print(f"      Failed to save snippet: {e}")
                return False
        return True

    def get_audio_transcript(self, start_t: float, end_t: float, step_id: str) -> str:
        duration = end_t - start_t
        if duration < 0.5:
            return ""

        audio_path = f"temp_step_{step_id}.mp3"
        try:
            subclip = self.video.subclip(start_t, end_t)
            subclip.audio.write_audiofile(audio_path, verbose=False, logger=None, bitrate="32k")

            result = self.model.transcribe(audio_path, fp16=True, language="en",
                                            initial_prompt=CLEAN_PROMPT)
            text = result["text"].strip()
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return text
        except Exception as e:
            print(f"Audio error: {e}")
            return ""

    def get_visual_status(self, start_frame: int, end_frame: int) -> str:
        mid_frame = int((start_frame + end_frame) / 2)
        frame_col = "Frame" if "Frame" in self.keypoints.columns else self.keypoints.columns[0]

        rows = self.keypoints[self.keypoints[frame_col] == mid_frame]
        doc = rows[rows["Role"] == "Doctor"]
        pat = rows[rows["Role"] == "Patient"]

        if not doc.empty and not pat.empty:
            return predict_zone_robust(doc.iloc[0], pat.iloc[0])
        return "No Data"

    def grade_student(self) -> list:
        print(f"\n{'STEP':<35} | {'AUDIO CHECK':<25} | {'VISUAL CHECK':<25} | {'RESULT'}")
        print("-" * 100)

        def natural_key(x):
            return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", str(x))]

        sorted_rows = self.timings.sort_values(by="step_id", key=lambda col: col.map(natural_key))

        score_log = []
        for _, row in sorted_rows.iterrows():
            step_id = str(row["step_id"])
            if step_id not in self.rubric:
                continue

            rule = self.rubric[step_id]
            step_name = rule["step_name"]
            modality = rule["modality"]

            start_f, end_f = row["start_frame"], row["end_frame"]
            start_t, end_t = start_f / self.fps, end_f / self.fps

            snippet_saved = False
            if step_id in self.snippet_targets and self.output_snippet_dir:
                fname = f"Step_{step_id}_{step_name}.mp4"
                snippet_saved = self.save_video_snippet(start_t, end_t, fname)

            # Audio check
            audio_status = "N/A"
            transcript = ""
            if modality in ["audio_dominant", "hybrid_dominant"]:
                transcript = self.get_audio_transcript(start_t, end_t, step_id)
                required_kws = rule.get("highlighted_keywords", [])

                if not required_kws:
                    audio_status = "OK (No keywords)"
                else:
                    hits = [k for k in required_kws if k.lower() in transcript.lower()]
                    audio_status = f"Match: {hits[0]}" if hits else f"Missed: {required_kws[0]}..."

            # Visual check
            visual_status = "N/A"
            if modality in ["video_dominant", "hybrid_dominant"]:
                detected_state = self.get_visual_status(start_f, end_f)
                visual_rules = rule.get("visual", {})
                target_exam = visual_rules.get("exam", "N/A")
                should_be_examining = target_exam not in ("N/A", "")

                if should_be_examining:
                    visual_status = (f"OK: {detected_state}" if "Examining" in detected_state
                                      else f"WARN Saw {detected_state}")
                else:
                    visual_status = f"OK: {detected_state}"

            passed = "Missed" not in audio_status
            grade = "PASS" if passed else "FAIL"

            name_display = step_name[:30] + (" [snippet]" if snippet_saved else "")
            print(f"{name_display:<35} | {audio_status[:25]:<25} | {visual_status[:25]:<25} | {grade}")
            score_log.append({"step": step_name, "grade": grade, "transcript": transcript})

        if self.output_snippet_dir:
            print(f"\nVerification snippets saved to: {self.output_snippet_dir}")

        return score_log
