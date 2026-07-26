"""
Action library / reference dictionary building: audio transcription + keyword
extraction + geometric zone-prediction heuristics, merged into a master
per-step dictionary.

Ported from FS_model.ipynb. FLAGGED PER USER FEEDBACK: the geometric
chest/torso zone-classification approach (`analyze_spatial_zone` /
`get_spatial_target` below) has not performed well and is a known-weak
piece of this pipeline -- kept in this pass per "port everything unique
and working first, judge later," but expect this to be cut or replaced.

Two sub-systems existed in the source notebook for closely related but not
identical purposes; both are ported:

1. `build_master_dictionary` (cell idx 59, the GPU-optimized/latest of four
   near-identical iterations at idx 55/56/59/71) -- does audio transcription
   (Whisper "base") AND geometric zone-prediction from keypoints in a single
   pass, per action_dictionary snippet folder.

2. `TranscriptGenerator` + `get_video_dominant_steps` + `update_modalities` +
   `update_dataset` + `create_master_dictionary` (cells idx 74-78) -- a later,
   more complete, CONFIRMED end-to-end successful chain (idx 74-78 all ran
   cleanly). Uses a larger Whisper "medium" model with a medical-vocabulary
   priming prompt and a hand-built per-step keyword map, and merges in a
   separate `step_view_labels.json` for "visual grading rules" -- NOTE that
   file is not produced by anything in this pipeline; it appears to be the
   output of the manual ZoningVisualizer labeling widget, which was
   deliberately excluded from this port. `create_master_dictionary()` will
   run fine without it (empty visual rules) but won't have real visual
   grading rules unless that JSON already exists on Drive from a prior
   manual labeling session, or is produced some other way.
"""
import glob
import json
import os
import re
import string
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

# ==========================================
# Sub-system 1: audio + spatial zone-prediction (idx 59)
# ==========================================
ZONES = {
    "HEAD_NECK": ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear"],
    "CHEST": ["L_Shoulder", "R_Shoulder", "L_Hip", "R_Hip"],
    "LEGS": ["L_Knee", "R_Knee", "L_Ankle", "R_Ankle"],
    "EXTREMITIES": ["L_Wrist", "R_Wrist"],
}


def _stop_words() -> set:
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return {
            "i", "me", "my", "myself", "we", "our", "you", "your", "the", "a", "an",
            "and", "or", "is", "are", "was", "were", "be", "have", "do", "to", "of",
            "in", "for", "with", "on", "at", "by", "from",
        }


STOP_WORDS = _stop_words()


def install_whisper() -> None:
    try:
        import whisper  # noqa: F401
    except ImportError:
        print("Installing OpenAI Whisper...")
        subprocess.run(["pip", "install", "git+https://github.com/openai/whisper.git"], check=True)


def extract_audio(video_path: str) -> str:
    wav_path = video_path.replace(".mp4", ".wav")
    if os.path.exists(wav_path):
        os.remove(wav_path)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", "-loglevel", "error", wav_path]
    subprocess.run(cmd)
    return wav_path


def extract_keywords(text: str) -> list:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return list(set(words))


def load_keypoints(kp_paths: dict) -> dict:
    print("Loading keypoint data...")
    kp_data = {}
    for angle, path in kp_paths.items():
        if os.path.exists(path):
            try:
                kp_data[angle] = pd.read_csv(path)
            except Exception:
                pass
    return kp_data


def get_distance(pt1, pt2) -> float:
    return float(np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))


def analyze_spatial_zone(doctor, patient) -> "str | None":
    """Nearest-centroid zone classification. Known-weak -- see module docstring."""
    doc_hands = []
    if pd.notna(doctor["L_Wrist_X"]):
        doc_hands.append((doctor["L_Wrist_X"], doctor["L_Wrist_Y"]))
    if pd.notna(doctor["R_Wrist_X"]):
        doc_hands.append((doctor["R_Wrist_X"], doctor["R_Wrist_Y"]))
    if not doc_hands:
        return None

    min_dist = float("inf")
    closest_zone = "UNKNOWN"

    for zone_name, kps in ZONES.items():
        pts = [(patient[f"{k}_X"], patient[f"{k}_Y"]) for k in kps if pd.notna(patient[f"{k}_X"])]
        if pts:
            center = np.mean(pts, axis=0)
            for hand in doc_hands:
                d = get_distance(hand, center)
                if d < min_dist:
                    min_dist = d
                    closest_zone = zone_name
    return closest_zone


def get_spatial_target(step_id: str, df_ann: pd.DataFrame, kp_data: dict):
    row = df_ann[df_ann["step_id"].astype(str) == step_id]
    if row.empty:
        row = df_ann[df_ann["step_id"].astype(str) == step_id.replace("_", ".")]
    if row.empty:
        return "UNKNOWN", 0.0

    start, end = int(row.iloc[0]["start_frame"]), int(row.iloc[0]["end_frame"])
    zone_votes = []

    for angle, df in kp_data.items():
        mask = (df["Frame"] >= start) & (df["Frame"] <= end)
        step_df = df[mask]
        if step_df.empty:
            continue

        for _, grp in step_df.groupby("Frame"):
            doc = grp[grp["Role"] == "Doctor"]
            pat = grp[grp["Role"] == "Patient"]
            if not doc.empty and not pat.empty:
                zone = analyze_spatial_zone(doc.iloc[0], pat.iloc[0])
                if zone:
                    zone_votes.append(zone)

    if zone_votes:
        top = Counter(zone_votes).most_common(1)[0]
        return top[0], round(top[1] / len(zone_votes), 2)
    return "NONE", 0.0


def build_master_dictionary(dataset_root: str, labels_path: str, annotations_path: str,
                             kp_paths: dict, output_path: str) -> None:
    """Audio transcription (Whisper "base", GPU if available) + spatial zone-prediction,
    per action-dictionary snippet folder."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware acceleration: {device.upper()}")
    if device == "cpu":
        print("Warning: running on CPU. Enable GPU for speed.")

    if not os.path.exists(labels_path):
        print("Labels file not found!")
        return
    with open(labels_path, "r") as f:
        manual_labels = json.load(f)

    df_ann = pd.read_csv(annotations_path) if os.path.exists(annotations_path) else pd.DataFrame()
    kp_data = load_keypoints(kp_paths)

    install_whisper()
    import whisper
    print(f"Loading Whisper model on {device.upper()}...")
    model = whisper.load_model("base", device=device)

    master_dict = {}
    all_folders = sorted(glob.glob(os.path.join(dataset_root, "step_*")))
    print(f"Processing {len(all_folders)} entries...")

    for folder in all_folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split("_")
        step_id = f"{parts[1]}_{parts[2]}" if len(parts) > 2 and parts[2].isdigit() else parts[1]

        modality = manual_labels.get(step_id, "video_dominant")

        entry = {
            "id": step_id,
            "folder": folder_name,
            "modality": modality,
            "paths": {
                "front": os.path.join(folder, "front.mp4"),
                "side": os.path.join(folder, "side.mp4"),
                "back": os.path.join(folder, "back.mp4"),
            },
            "transcript": None,
            "keywords": [],
            "spatial_target": "UNKNOWN",
            "spatial_confidence": 0.0,
        }

        if "audio" in modality or "hybrid" in modality:
            audio_source = entry["paths"]["side"] if os.path.exists(entry["paths"]["side"]) else entry["paths"]["front"]
            if os.path.exists(audio_source):
                try:
                    wav = extract_audio(audio_source)
                    use_fp16 = device == "cuda"
                    res = model.transcribe(wav, language="en", fp16=use_fp16)
                    text = res["text"].strip()
                    if text.lower() in ["you", "thank you", "subtitles by ...", ""]:
                        text = ""
                    entry["transcript"] = text
                    entry["keywords"] = extract_keywords(text)
                    if os.path.exists(wav):
                        os.remove(wav)
                    if text:
                        print(f"   [{step_id}] Text: \"{text[:40]}...\"")
                except Exception as e:
                    print(f"   Audio error: {e}")

        if "video" in modality or "hybrid" in modality:
            if not df_ann.empty and kp_data:
                target, conf = get_spatial_target(step_id, df_ann, kp_data)
                entry["spatial_target"] = target
                entry["spatial_confidence"] = conf
                if conf > 0.5:
                    print(f"   [{step_id}] Zone: {target} ({conf:.2f})")

        master_dict[step_id] = entry

    with open(output_path, "w") as f:
        json.dump(master_dict, f, indent=4)
    print(f"\nMaster dictionary built: {output_path}")


# ==========================================
# Sub-system 2: transcript generation + keyword highlighting + merge (idx 74-78)
# CONFIRMED end-to-end successful -- prefer this chain over sub-system 1's audio path.
# ==========================================
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

STEP_KEYWORD_MAP = {
    "Greet": ["name is", "address you", "cardiovascular exam"],
    "Consent": ["permission", "examine", "okay"],
    "Position": ["45 degrees", "comfortable"],
    "Expose": ["expose", "chest"],
    "General_Inspection": ["foot of the bed", "general inspection", "distress", "alert"],
    "Inspect_Syndromic": ["syndromic", "marfan", "turner", "down"],
    "Inspect_Anterior_Chest": ["scars", "deformities", "visible pulsation", "apex beat", "raise your arms"],
    "Inspect_Hands": ["hands", "clubbing", "cyanosis", "splinter", "janeway", "osler", "tar staining"],
    "Inspect_Forearms": ["forearms", "needle marks", "antecubital fossa", "xanthoma"],
    "Palpate_Radial": ["pulse", "rate", "rhythm"],
    "Radial_Symmetry": ["radio-radial", "delay"],
    "Femoral": ["femoral", "groin"],
    "Collapsing": ["shoulder", "pain", "raise your arm", "collapsing"],
    "Inspect_Face": ["eyes", "mouth", "tongue", "jaundice", "pallor", "malar", "cyanosis", "high arch"],
    "Inspect_Neck": ["neck", "look up", "turn"],
    "Inspect_JVP": ["jvp", "jugular", "root of the neck"],
    "Abdominojugular": ["tummy", "abdomen", "press", "reflux"],
    "Carotids": ["bruit", "hold your breath"],
    "Apex": ["apex", "beat", "feel", "character"],
    "Heave": ["heave", "parasternal", "lift"],
    "Thrill": ["thrill", "palpable"],
    "Auscultate": ["listen", "heart sounds"],
    "Mitral": ["mitral", "roll", "left"],
    "Turn_Left": ["turn", "left", "side"],
    "Breathing_Maneuver": ["breathe in", "breathe out", "hold"],
    "Turn_Back": ["turn back", "flat"],
    "Sit_Up": ["sit up", "lean forward"],
    "Aortic_Regurgitation": ["breathe out", "hold", "lean"],
    "Inspect_Back": ["back", "spine", "sacral"],
    "Lung_Bases": ["lungs", "deep breath", "in and out"],
    "Inspect_Legs": ["legs", "edema", "press", "swelling", "pain"],
    "Thank": ["thank", "dressed", "cover", "end of"],
    "Summary": ["summary", "findings", "vital", "temperature", "urine"],
}


class TranscriptGenerator:
    """GPU Whisper ("medium") transcription primed with cardiovascular-exam vocabulary."""

    def __init__(self, action_dict_root: str, modality_json_path: str,
                 output_json_path: str, whisper_model_size: str = "medium",
                 skip_video_dominant: bool = True):
        import whisper

        self.action_dict_root = action_dict_root
        self.modality_json_path = modality_json_path
        self.output_json_path = output_json_path
        self.skip_video_dominant = skip_video_dominant

        print("\nInitializing Transcript Generator (GPU mode)...")
        print(f"   Loading Whisper '{whisper_model_size}' model to GPU...")
        self.model = whisper.load_model(whisper_model_size, device="cuda")
        print("   Model loaded.")

        self.context_prompt = self._build_medical_primer()
        self.keywords_list = self._build_keyword_list()
        self.modality_map = self._load_modality_map()
        self.master_data = []
        print(f"   -> Primed with medical vocabulary ({len(self.keywords_list)} keywords tracked).")

    def _build_medical_primer(self) -> str:
        return " ".join(CHECKLIST_TEXT.split())

    def _build_keyword_list(self) -> list:
        raw = CHECKLIST_TEXT.replace("Cardiovascular examination context. Key terms:", "")
        raw = raw.replace("\n", " ")
        return [k.strip().lower() for k in raw.split(",") if k.strip()]

    def _load_modality_map(self) -> dict:
        if os.path.exists(self.modality_json_path):
            with open(self.modality_json_path, "r") as f:
                return json.load(f)
        return {}

    def get_step_info(self, folder_name: str):
        match = re.search(r"step_(\d+(?:_\d+)?)_(.*)", folder_name)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def extract_keywords(self, transcript: str) -> list:
        text_lower = transcript.lower()
        return list({kw for kw in self.keywords_list if kw in text_lower})

    def extract_audio_from_video(self, video_path: str, audio_output_path: str) -> bool:
        import moviepy.editor as mp
        try:
            video = mp.VideoFileClip(video_path)
            if video.audio is None:
                video.close()
                return False
            video.audio.write_audiofile(audio_output_path, verbose=False, logger=None, bitrate="32k")
            video.close()
            return True
        except Exception as e:
            print(f"      Audio error: {e}")
            return False

    def transcribe_file(self, audio_path: str) -> str:
        result = self.model.transcribe(
            audio_path, initial_prompt=self.context_prompt, fp16=True, language="en",
        )
        return result["text"].strip()

    def process_dataset(self) -> None:
        step_folders = sorted(glob.glob(os.path.join(self.action_dict_root, "step_*")))
        print(f"\nFound {len(step_folders)} step folders.\n")

        count = 0
        skipped = 0

        for folder in step_folders:
            folder_name = os.path.basename(folder)
            step_id, step_name = self.get_step_info(folder_name)
            if not step_id:
                continue

            modality = self.modality_map.get(step_id, "unknown")
            if self.skip_video_dominant and modality == "video_dominant":
                skipped += 1
                continue

            print(f"Processing: {step_name} (ID: {step_id})")

            video_path = os.path.join(folder, "side.mp4")
            if not os.path.exists(video_path):
                cands = glob.glob(os.path.join(folder, "*.mp4"))
                if cands:
                    video_path = cands[0]
                else:
                    print("      No video file found.")
                    continue

            audio_path = os.path.join(folder, "temp_audio.mp3")
            txt_path = os.path.join(folder, "transcript.txt")

            if self.extract_audio_from_video(video_path, audio_path):
                transcript_text = self.transcribe_file(audio_path)
                keywords_found = self.extract_keywords(transcript_text)

                with open(txt_path, "w") as f:
                    f.write(transcript_text)

                self.master_data.append({
                    "step_id": step_id,
                    "step_name": step_name,
                    "modality": modality,
                    "transcript_text": transcript_text,
                    "highlighted_keywords": keywords_found,
                })

                if os.path.exists(audio_path):
                    os.remove(audio_path)
                count += 1
                print(f"      Transcribed ({len(transcript_text.split())} words)")

        with open(self.output_json_path, "w") as f:
            json.dump(self.master_data, f, indent=4)

        print(f"\nProcessed {count} steps. Skipped {skipped} video-dominant steps.")
        print(f"Master JSON saved to: {self.output_json_path}")


def get_video_dominant_steps(modality_json_path: str) -> None:
    if not os.path.exists(modality_json_path):
        print("Error: modality labels file not found.")
        return

    with open(modality_json_path, "r") as f:
        data = json.load(f)

    print("\nVideo-dominant steps (audio ignored):")
    print("-" * 40)
    sorted_ids = sorted(data.keys(), key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)])
    count = 0
    for step_id in sorted_ids:
        if data[step_id] == "video_dominant":
            print(f"   - Step {step_id}")
            count += 1
    print("-" * 40)
    print(f"Total video-dominant steps: {count}")


def update_modalities(transcripts_path: str, modality_labels_path: str) -> None:
    print("Updating modalities...")

    if not os.path.exists(modality_labels_path):
        print(f"Error: modality file not found at {modality_labels_path}")
        return
    with open(modality_labels_path, "r") as f:
        modality_map = json.load(f)
    print(f"   -> Loaded {len(modality_map)} modality labels.")

    if not os.path.exists(transcripts_path):
        print(f"Error: transcripts file not found at {transcripts_path}")
        return
    with open(transcripts_path, "r") as f:
        transcripts = json.load(f)
    print(f"   -> Loaded {len(transcripts)} transcripts.")

    updated_count = 0
    missing_count = 0

    for entry in transcripts:
        step_id = str(entry.get("step_id", ""))
        if step_id in modality_map:
            new_modality = modality_map[step_id]
            if entry.get("modality") != new_modality:
                entry["modality"] = new_modality
                updated_count += 1
        else:
            missing_count += 1

    if updated_count > 0:
        with open(transcripts_path, "w") as f:
            json.dump(transcripts, f, indent=4)
        print(f"\nUpdated {updated_count} transcripts with correct modalities.")
    else:
        print("\nNo changes needed.")

    if missing_count > 0:
        print(f"Warning: {missing_count} steps had no matching modality label.")


def update_dataset(transcripts_path: str, modality_labels_path: str) -> None:
    """Re-derives `highlighted_keywords` per step using STEP_KEYWORD_MAP (smarter than
    the flat keyword-scan in extract_keywords)."""
    print("Updating dataset...")

    if not os.path.exists(transcripts_path):
        print(f"Error: transcript file missing at {transcripts_path}")
        return
    if not os.path.exists(modality_labels_path):
        print(f"Error: modality file missing at {modality_labels_path}")
        return

    with open(transcripts_path, "r") as f:
        transcripts = json.load(f)
    with open(modality_labels_path, "r") as f:
        modality_map = json.load(f)

    print(f"   -> Loaded {len(transcripts)} transcripts and {len(modality_map)} modality labels.")

    updated_count = 0
    for entry in transcripts:
        step_id = str(entry.get("step_id", ""))
        step_name = entry.get("step_name", "")
        text_lower = entry.get("transcript_text", "").lower()

        if step_id in modality_map:
            entry["modality"] = modality_map[step_id]

        found_keywords = []
        for key, keywords in STEP_KEYWORD_MAP.items():
            if key in step_name:
                for kw in keywords:
                    if kw in text_lower:
                        found_keywords.append(kw)

        entry["highlighted_keywords"] = sorted(set(found_keywords))
        updated_count += 1

    with open(transcripts_path, "w") as f:
        json.dump(transcripts, f, indent=4)

    print(f"\nUpdated {updated_count} steps with modalities & smart keywords.")
    print(f"Saved to: {transcripts_path}")


def _natural_sort_key(text: str):
    return re.split(r"(\d+)", text)


def create_master_dictionary(transcripts_path: str, modality_path: str,
                              view_labels_path: str, output_path: str) -> None:
    """Merges transcripts + modality labels + (optional) geometric view rules
    into the final action_master_dictionary.json.

    `view_labels_path` is NOT produced anywhere in this pipeline (see module
    docstring) -- if missing, visual grading rules are simply left empty.
    """
    print("Building action master dictionary...")

    if os.path.exists(transcripts_path):
        with open(transcripts_path, "r") as f:
            transcripts_list = json.load(f)
        print(f"   -> Loaded {len(transcripts_list)} transcript entries.")
    else:
        print(f"Critical: transcripts not found at {transcripts_path}")
        return

    if os.path.exists(modality_path):
        with open(modality_path, "r") as f:
            modality_map = json.load(f)
        print(f"   -> Loaded {len(modality_map)} modality labels.")
    else:
        modality_map = {}
        print("Warning: modality labels not found.")

    if os.path.exists(view_labels_path):
        with open(view_labels_path, "r") as f:
            view_rules_map = json.load(f)
        print(f"   -> Loaded {len(view_rules_map)} geometric view rules.")
    else:
        view_rules_map = {}
        print("Warning: view labels not found (expected if not manually labeled).")

    master_dict = {}

    def clean_id(id_val) -> str:
        return str(id_val).strip()

    for entry in transcripts_list:
        step_id = clean_id(entry.get("step_id"))
        master_dict[step_id] = {
            "step_name": entry.get("step_name", "Unknown"),
            "transcript_text": entry.get("transcript_text", ""),
            "highlighted_keywords": entry.get("highlighted_keywords", []),
            "modality": entry.get("modality", "unknown"),
            "grading_rules": {"visual": {}, "audio": {}},
        }

    for step_id, modality in modality_map.items():
        step_id = clean_id(step_id)
        if step_id in master_dict:
            master_dict[step_id]["modality"] = modality
        else:
            master_dict[step_id] = {
                "step_name": f"Step_{step_id}",
                "transcript_text": "",
                "highlighted_keywords": [],
                "modality": modality,
                "grading_rules": {"visual": {}, "audio": {}},
            }

    for step_id, rules in view_rules_map.items():
        step_id = clean_id(step_id)
        if step_id in master_dict:
            master_dict[step_id]["grading_rules"]["visual"] = rules
        else:
            master_dict[step_id] = {
                "step_name": f"Step_{step_id}",
                "transcript_text": "",
                "highlighted_keywords": [],
                "modality": "unknown",
                "grading_rules": {"visual": rules, "audio": {}},
            }

    sorted_keys = sorted(master_dict.keys(), key=_natural_sort_key)
    sorted_dict = {k: master_dict[k] for k in sorted_keys}

    with open(output_path, "w") as f:
        json.dump(sorted_dict, f, indent=4)

    print(f"\nMaster dictionary saved to: {output_path}")
    print(f"Total steps: {len(master_dict)}")
