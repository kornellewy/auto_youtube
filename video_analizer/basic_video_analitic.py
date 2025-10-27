"""
lllm sugesion :
Dla naprawdę dobrego filmu YouTube (wiadomość jasna, montaż dynamiczny, hook w pierwszych 3 s) mapa cieplna musi wyglądać tak:

    0-3 s – jedno, ściśle skupione białe/plomieniste jądro
        na twarzy/nagłówku prowadzącego albo na głównym obiekcie w 1/3 górnej (tzw. „eye-level rule”).
        zero rozlanego czerwonego tła – inaczej widz szuka „gdzie patrzeć” i odpada.
    3-10 s – jądro przesuwa się wraz z:
        ruchem warg (szybki cut na close-up),
        pojawieniem się tekstu/title (kontrast, duża czcionka) – wtedy dwa jasne punkty: twarz + napis, ale twarz wciąż dominuje (większa jasność na mapie).
    10-30 s – sekwencyjne bliki:
        cut → nowy shot → nowy jasny punkt dokładnie tam, gdzie twórca chce klik “Subscribe” lub zachętę interakcyjną.
        Każdy kolejny kadr ma 1 jasne centrum; reszta ekranu jest granatowo-zielona (niskie saliency).

Jak to sprawdzić u siebie
Załaduj .npy z mojej funkcji:
Python
Copy

mean = np.load("scene_002_mean_saliency.npy")
print("Focus %:", (mean > 0.7).mean() * 100)   # >70 % pikseli = mocno skupione

    > 60 % pikseli o wartości > 0.7 = perfekcyjny hook.
    < 30 % albo rozlewające się czerwone tło = słabe zatrzymanie uwagi – przycinaj kadry, zmniejsz tło, przybliż twarz/tekst.

Podsumowanie – „definitywna” odpowiedź
Dobra mapa YouTube’a to łańcuch jasnych, drobnych, przemieszczających się jąder (twarz → tekst → produkt/subskrypcja), każde w 1/3 górnej, reszta ekranu ciemna na mapie.
Jeśli widzisz jedno, skupione białe/plomieniste centrum przez pierwsze 3 sekundy – trafiłeś w psychologiczny hook; jeśli czerwień się rozlewa – kadr wymaga kadrowania albo lepszej hierarchii wizualnej.


Dynamika sceny w dobrym filmie YouTube
(wartość „avg_diff” z Twojego pipeline’a – średnia różnica pikselowej normy L1 między klatkami)
1. Pierwsze 3 s (hook)

    avg_diff 8-15
    maksymalny skok w 1-szej klatce po cut-cie (cięcie, zoom-in, title smash)
    potem natychmiast spadek do 3-5 żeby nie „męczyć” oka

2. 3-10 s (rozwinięcie)

    avg_diff 3-6
    lekko podniesiony poziom = kamera się porusza, gestykulacja, ale bez szybkich flash-cutów
    rytm: pik co ~0,7-1,0 s (kolejne uderzenie gestem/palcem w kamerę)

3. 10-30 s (value delivery)

    avg_diff 2-4
    najspokojniejszy fragment – widz łapie informację
    pojedyncze spiki do 6-8 tylko gdy pokazujesz:
        nowy obiekt
        screen / statystyka
        zmiana planu (wide → tight)

4. 30-45 s (reset + re-hook)

    mini-peaks avg_diff 7-12 co ~8-10 s
    cel: ponownie obudzić scroll-phone viewers
    forma: jump-cut, sound whoosh, text pop, speed-ramp 110 %

5. Cała scena 45-90 s

    średnia ruchu (avg_diff) 4-7
    std-dev ~2-3 – regularne, przewidywalne „fale” zamiast chaotycznego szumu
    żadnych długich (>2 s) odcinków z avg_diff <1 – wtedy TikTok/YouTube Shorts liczy jako „nuda” i spada reach


Perfect scene length for a good YouTube video
Table
Copy
Moment	Ideal length	What happens
HOOK (cut 1)	0 – 3 s	One hard cut or zoom; max visual spike avg_diff ≈ 8-15
PAY-OFF	3 – 10 s	Single continuous shot or 2-3 rapid cuts; avg_diff 3-6
VALUE DELIVERY	10 – 30 s	Calmest segment; ≤ 5 cuts, avg_diff 2-4
MINI-RESET	30 – 45 s	New spike avg_diff 7-12 (text pop, B-roll, sound whoosh)
FULL SCENE total	7 – 12 s mean duration	Keeps audience > 50 % retention at 4-min mark
MAXIMUM for one topic	≤ 20 s	Longer only if story is hyper-engaging

    If your scene-average avg_diff is < 2 for > 2 s – speed it up or delete.
    If > 10 for > 3 s straight – add breathing space or viewers bounce.

Stick to 7-12 s per scene, spike every 8-10 s, and YouTube’s algorithm will reward you with longer watch-time.

{"meta": {"num_scenes": 118, "total_duration": 684.375, "avg_scene_duration": 5.79978813559322, "std_scene_duration": 4.374035602863045, "tanscript": " "}}
"""

from __future__ import annotations
import os
import time
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import ffmpeg
import whisper_timestamped as whisper
import librosa

# import soundfile as sf
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

gemini_pricing = {
    "gemini-2.5-flash": {
        "input_tokens_cost": 0.30,  # for text/image/video
        "output_tokens_cost": 2.50,
    },
    "gemini-2.5-flash-lite": {
        "input_tokens_cost": 0.30,  # for text/image/video
        "output_tokens_cost": 0.40,
    },
    "gemini-2.0-flash": {
        "input_tokens_cost": 0.10,  # for text/image/video
        "output_tokens_cost": 0.40,
    },
    "gemini-2.0-flash-lite": {
        "input_tokens_cost": 0.075,
        "output_tokens_cost": 0.30,
    },
}


class VideoInspector:
    """
    End-to-end analyser:
        1.  Split video into scenes (PySceneDetect)
        2.  Compute frame-difference dynamics
        3.  Generate saliency heat-maps
        4.  Evaluate each scene with Gemini-2.0-Flash-Lite
        5.  Aggregate meta-data & final markdown report
    """

    GEMINI_MODEL = "gemini-2.5-flash-lite"
    FINAL_REPORT_MODEL = "gemini-2.5-flash"

    MEDIA_SCORE_PROMPT = """
        ROLE: You are a world-class data-driven media analyst, specializing in the predictive performance of short-form video content on platforms like TikTok and YouTube Shorts. Your analysis is grounded in a quantitative framework that evaluates a video's potential to capture and sustain audience attention, leading to algorithmic success.

        TASK: Conduct a comprehensive, data-driven analysis of the provided media scene based on the following inputs:

            1. A representative key-frame image.
            2. A frame-dynamics plot (visualizing motion and change as `avg_diff`).
            3. The complete audio track for the scene.
            4. The scene's dialogue transcript.

        Your task is to score the scene's visual and audio components, its potential to retain viewers, and the sensitivity of its topic. All scores must be integers on a scale of 0 to 100.

        ---
        ### SCORING CRITERIA
        ---

        #### Part 1: Visual Analysis & Retention Potential

        1.  **video_viewer_engagement (Hook Potential)**: An assessment of the video's ability to capture attention in the first 3 seconds (i.e., its "Hook Rate" potential).
            *   **Scoring Rubric**:
                *   `90-100` (Top Tier): A single, high-contrast focal point (e.g., face, bold text) in the upper third. Poses a direct question or shows a surprising action. `avg_diff` plot shows a sharp spike (8-15) on the first cut. Predicts a Hook Rate >30%.
                *   `70-89` (Strong): Clear focal point, but may have minor competing elements. Opening is intriguing but not startling. Predicts a Hook Rate of 20-25%.
                *   `0-69` (Poor): Visually confusing, slow, or generic opening. Highly likely to be scrolled past.

        2.  **video_retention_score (Hold Potential)**: An assessment of the clip's ability to sustain user attention after the initial hook (i.e., its "Hold Rate" potential).
            *   **Scoring Rubric**:
                *   `90-100` (Exceptional): Delivers on the hook's promise immediately. Dynamic pacing introduces new visual information (zooms, B-roll, overlays) every 1.5-2 seconds. Predicts a Hold Rate >50%.
                *   `70-89` (Good): Pacing is generally strong, but may have moments longer than 2 seconds without a visual change. Narrative is clear and engaging. Predicts a Hold Rate of 40-50%.
                *   `0-69` (Poor): Fails to deliver on the hook. Pacing is static, content is uninteresting, causing a steep decline in viewership.

        3.  **video_emotional_impact**: The visual's ability to evoke a specific, intended emotion (e.g., joy, humor, empathy, surprise) through imagery alone.
        4.  **video_content_clarity**: How quickly and easily a viewer can understand the scene's subject, action, and context from the visuals. High scores are for scenes that are immediately comprehensible.
        5.  **video_visual_hierarchy**: The effectiveness of the composition in guiding the viewer's eye to the most important elements. A high score means the primary subject is the clear focal point.
        6.  **video_text_readability**: The legibility of any on-screen text, considering font, size, color, and contrast. If no text is present, the value for this key must be `null`.

        #### Part 2: Audio Analysis

        7.  **audio_technical_quality**: The technical fidelity of the recording and mix. A high score indicates clean, well-balanced audio with no audible flaws (hiss, clipping, etc.).
        8.  **audio_voice_clarity**: The clarity and intelligibility of spoken words against other audio elements (music, SFX). A high score means dialogue is crisp and easily understood.
        9.  **audio_emotional_impact**: The audio's effectiveness in creating a specific mood or emotional response that complements the scene's visual emotional impact.
        10. **audio_content_relevance**: How well the audio choice (Original, Commercial, or Trending) serves the video's strategic goal. High scores mean the choice is purposeful and aligns with the brand and content type.
        11. **audio_visual_match**: The synergy and timing between the audio track and the visual elements. A high score means the sound perfectly complements and enhances the on-screen action and pacing.

        #### Part 3: Topic Sensitivity Analysis

        12. **topic_sensitivity_score**: An assessment of the content's risk of being flagged or violating platform community guidelines, based on its subject matter.
            *   **Scoring Rubric**:
                *   `0-20` (Low): Neutral, safe topics (e.g., tutorials, product reviews).
                *   `21-50` (Moderate): Touches on personal opinions but is not widely polarizing (e.g., lifestyle choices, hobbies).
                *   `51-80` (High): Emotionally charged or controversial topics (e.g., politics, social issues) that may require an EDSA context to be safe.
                *   `81-100` (Extreme): Highly personal, traumatic, or extremely polarizing subjects (e.g., mental health crises, violence). High risk of removal.

        ---
        ### OUTPUT FORMAT
        ---
        You must reply **only** with a single, valid JSON object. Do not include any explanatory text, markdown formatting, or code fences.

        Example Response:
        ```json
        {
            "video_viewer_engagement": 85,
            "video_emotional_impact": 70,
            "video_content_clarity": 95,
            "video_visual_hierarchy": 88,
            "video_text_readability": 100,
            "video_retention_score": 92,
            "audio_voice_clarity": 90,
            "audio_emotional_impact": 75,
            "audio_content_relevance": 95,
            "audio_technical_quality": 85,
            "audio_visual_match": 92,
            "topic_sensitivity_score": 45
        }
        """

    FINAL_REPORT_PROMPT = """
ROLE: You are "AI-Critic," a world-class YouTube video analyst and strategist. Your expertise lies in interpreting video performance data and translating it into a concise, actionable scorecard. You understand that success on modern platforms is driven by a deep understanding of algorithmic rewards for dynamic pacing and the psychology of viewers with short attention spans. Your advice is direct, insightful, and always aimed at helping creators make tangible, data-informed improvements to their content.

TASK: Generate a comprehensive "YouTube Video Score-Card" based on the provided single, flat JSON object containing the overall metadata for a video. The report must be clear, data-driven, and provide specific, actionable advice grounded in the principles of algorithmic favorability. You will populate a predefined report template with the data from the JSON, interpreting the scores to provide grades and dynamically generate lists of improvements and expected outcomes.

INPUT: You will receive a single, flat JSON object. Example:
{
"video_viewer_engagement": 68,
"video_emotional_impact": 60,
"video_content_clarity": 95,
"video_visual_hierarchy": 75,
"video_text_readability": 100,
"video_retention_score": 65,
"audio_voice_clarity": 95,
"audio_emotional_impact": 70,
"audio_content_relevance": 90,
"audio_technical_quality": 80,
"audio_visual_match": 90,
"topic_sensitivity_score": 20
}

OUTPUT: Your output must be a markdown-formatted report that strictly follows the provided template. You will replace all placeholders like [[PLACEHOLDER]] with the corresponding data from the JSON input. For placeholders that require interpretation, you will use the provided logic.
Logic and Rules for Dynamic Content:

    Grading ([[grade(...)]]): 90-100: A+ | 80-89: A | 70-79: B | 60-69: C | Below 60: F.

    Safety Level & Action ([[safety_level]], [[safety_action]]):

        0-20: Low Sensitivity -> "Content is generally safe for all advertisers."

        21-50: Moderate Sensitivity -> "Review for sensitive themes. If content is educational, add a disclaimer (EDSA context) to mitigate risk before enabling all ad formats."

        51-80: High Sensitivity -> "Consider disabling comments or adding a clear disclaimer. Video may have limited monetization."

        81-100: Extreme Sensitivity -> "Strongly consider age-restricting the video and disabling monetization to avoid a channel strike."

    Diagnostic Statements:

        [[hook_diagnosis]]: Based on video_viewer_engagement. If > 85, state: "The video's opening is strong and effectively captures initial attention." If < 70, state: "The video's opening is weak and likely fails to stop viewers from scrolling." Otherwise, state: "The video's opening is average and could be strengthened to improve performance."

        [[retention_diagnosis]]: Based on video_retention_score. If > 85, state: "The content is highly engaging and successfully holds viewer interest after the hook." If < 70, state: "The content struggles to maintain viewer interest, indicating critical issues with pacing. The algorithm detects low narrative momentum." Otherwise, state: "The content holds attention reasonably well but has room for improvement in pacing to maximize algorithmic favorability."

    Dynamically Generated Lists & Statements (LLM must generate these):

        [[visual_improvement_list]]: Analyze all visual scores (video_viewer_engagement, video_emotional_impact, video_content_clarity, video_visual_hierarchy). Generate a numbered list of up to 10 actionable fixes, starting with the metric with the lowest score. For each, state the metric, its score, and 1-2 specific solutions based on the deep research (e.g., "Improve Video Emotional Impact (Score: 60) by adjusting color grading or adding contextual B-roll to better match the narrative's tone.").

        [[audio_improvement_list]]: Analyze all audio scores (audio_voice_clarity, audio_emotional_impact, audio_technical_quality). Generate a numbered list of up to 10 actionable fixes, starting with the metric with the lowest score. For each, state the metric, its score, and 1-2 specific solutions (e.g., "Improve Audio Emotional Impact (Score: 70) by A/B testing different tracks from the commercial library or adjusting the volume of background music to better support the voice-over."). Mention the opportunity to create "Original Audio" if scores are high, as 65% of users prefer it.

        [[quick_wins_list]]: Based on your entire analysis, generate a numbered list of the top 3-4 most critical and easy-to-implement "24-Hour Quick Wins." These should directly address the lowest-scoring metrics identified in the report. For example, if the hook is the weakest link, the first point must be about fixing the first 3 seconds.

        [[dynamic_impact_statement]]: Synthesize the overall findings to generate a realistic "Expected Impact" statement. If scores are very low, the potential percentage increase in retention should be high (e.g., +25-40%). If scores are mostly good, the potential increase should be lower (e.g., +5-10%). The statement must explain why this impact is expected, referencing the algorithm or viewer psychology.

The Report Template to Be Filled:
code Markdown
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
YOUTUBE VIDEO SCORE-CARD

*Generated by AI-Critic*

---

EXECUTIVE SUMMARY (0-100)

| Metric                    | Score                               | Grade                               |
| ------------------------- | ----------------------------------- | ----------------------------------- |
| **Hook Strength (0-3s)**  | [[video_viewer_engagement]]         | [[grade(video_viewer_engagement)]]  |
| **Hold Strength (Pacing)**| [[video_retention_score]]           | [[grade(video_retention_score)]]    |
| **Overall Audio Quality** | [[audio_technical_quality]]         | [[grade(audio_technical_quality)]]  |
| **Topic Safety**          | [[topic_sensitivity_score]]         | [[grade(topic_sensitivity_score)]]  |

---

**1. RETENTION ANALYSIS**

*   **Hook Diagnosis (First 3 Seconds):** [[hook_diagnosis]] (Score: [[video_viewer_engagement]])
*   **Retention Diagnosis (Content Body):** [[retention_diagnosis]] (Score: [[video_retention_score]])

**Primary Recommendation:**
To improve overall retention, focus on strengthening the weakest link. Based on the scores, the video is stronger at [[if video_viewer_engagement > video_retention_score then "hooking viewers than holding them. Focus on improving content pacing and value delivery." else "holding viewers than hooking them. Focus on creating a more impactful first 3 seconds."]]

**Pacing & Algorithm Tip:** To improve your Hold Strength, introduce a significant visual change (e.g., a zoom, B-roll clip, or text overlay) approximately every **1.5 seconds**. This signals high narrative momentum to the algorithm and keeps viewers engaged.

---

**2. VISUAL PERFORMANCE & IMPROVEMENT CHECKLIST**

[[visual_improvement_list]]

---

**3. AUDIO PERFORMANCE & IMPROVEMENT CHECKLIST**

[[audio_improvement_list]]

---

**4. TOPIC & MONETIZATION**

*   **Sensitivity Score:** [[topic_sensitivity_score]] / 100
*   **Level:** [[safety_level]]
*   **Recommended Action:** [[safety_action]]

---

**24-HOUR QUICK WINS**

[[quick_wins_list]]

---

**EXPECTED IMPACT**

[[dynamic_impact_statement]]


"""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        video_path: str | Path,
        work_dir: str | Path | None = None,
        api_keys: List[str] | None = None,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)

        # working folder
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="vi_"))
        self.work_dir.mkdir(exist_ok=True)

        # API key rotation
        self._api_keys = api_keys or [
            os.getenv(f"GOOGLE_API_KEY{i}") for i in range(1, 11)
        ]
        self._api_keys = [k for k in self._api_keys if k]
        if not self._api_keys:
            raise ValueError("No Gemini API keys found.")
        self._key_idx = 0
        self._configure_api()

        # storage
        self.scenes_csv = self.work_dir / "scenes.csv"
        self.images_dir = self.work_dir / "keyframes"
        self.plots_dir = self.work_dir / "dynamics"
        self.saliency_dir = self.work_dir / "saliency"
        self.scane_audio_dir = self.work_dir / "scane_audio"
        self.scane_audio_transcript_dir = self.work_dir / "scane_audio_transcript"
        self.frame_that_reprezent_scene_descpriton_dir = self.work_dir / "description"
        self.scane_audio_metric_dir = self.work_dir / "scene_audio_metric"
        self.visual_audio_match_path = self.work_dir / "visual_audio_match"
        for d in (
            self.images_dir,
            self.plots_dir,
            self.saliency_dir,
            self.frame_that_reprezent_scene_descpriton_dir,
            self.scane_audio_transcript_dir,
            self.scane_audio_dir,
            self.scane_audio_metric_dir,
            self.visual_audio_match_path,
        ):
            d.mkdir(exist_ok=True)

        # results
        self.scenes: List[Dict] = []
        self.input_tokens = {
            key: {"input_tokens": 0, "output_tokens": 0}
            for key in gemini_pricing.keys()
        }
        self.output_tokens = {
            key: {"input_tokens": 0, "output_tokens": 0}
            for key in gemini_pricing.keys()
        }

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------
    def analyse(self) -> Path:
        self.scenes: List[Dict] = []
        """Run full pipeline and return path to final markdown report."""
        print("[1/6] Detecting scenes…")
        self._detect_scenes()

        print("[2/6] Computing frame dynamics & plots…")
        self._compute_dynamics()

        # print("[3/6] Generating saliency maps…")
        # self._generate_saliency() # advence

        # print("[4/6] Scoring scenes with Gemini…")
        # self._score_scenes() #

        print("[5/7] Audio pipeline (transcript + quality + match)…")
        self._audio_pipeline()

        self._llm_description_of_scene()

        print("[5/6] Meta-analysis…")
        self._build_metadata()

        print("[6/6] Writing final report…")
        report_path = self._write_report()
        print(f"[OK] Report saved → {report_path}")

        print("[6/6] Writing costs")
        self.save_costs()

        return report_path

    # ------------------------------------------------------------------
    # 1. Scene detection
    # ------------------------------------------------------------------
    def _detect_scenes(self):
        if self.scenes_csv.exists():
            print("[Scene] Loading existing CSV …")
            self.scenes = []
            with open(self.scenes_csv, newline="") as f:
                for row in csv.DictReader(f):
                    self.scenes.append(
                        {
                            "idx": int(row["scene_idx"]),
                            "start_time": float(row["start_time"]),
                            "end_time": float(row["end_time"]),
                            "start_frame": int(row["start_frame"]),
                            "end_frame": int(row["end_frame"]),
                        }
                    )
            n = len(self.scenes)
            print(f"[Scene]  {n} scenes restored from CSV")

            #  make sure every key-frame exists  --------------------------
            need_frames = [
                i
                for i in range(n)
                if not (self.images_dir / f"scene_{i:03d}.jpg").exists()
            ]
            if need_frames:
                print(f"[Scene]  {len(need_frames)} key-frames missing – extracting …")
                self._extract_keyframes(list(range(len(self.scenes))))
            return
        else:
            video = open_video(str(self.video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            scene_manager.detect_scenes(video)
            scenes = scene_manager.get_scene_list()
            with open(self.scenes_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["scene_idx", "start_time", "end_time", "start_frame", "end_frame"]
                )
                for i, (start, end) in enumerate(scenes):
                    self.scenes.append(
                        {
                            "scene_idx": i,
                            "start_time": start.get_seconds(),
                            "end_time": end.get_seconds(),
                            "start_frame": start.get_frames(),
                            "end_frame": end.get_frames(),
                        }
                    )
                    writer.writerow(
                        [
                            i,
                            start.get_seconds(),
                            end.get_seconds(),
                            start.get_frames(),
                            end.get_frames(),
                        ]
                    )
            # save keyframes
            self._extract_keyframes(list(range(len(scenes))))

    def _extract_keyframes(self, scene_indices: list):
        """Extract sharpest frame for given scene indices."""
        cap = cv2.VideoCapture(str(self.video_path))
        bounds = [(s["start_frame"], s["end_frame"]) for s in self.scenes]
        n = len(self.scenes)

        best_sharpness = [-1] * n
        best_frame = [None] * n

        frame_idx = 0
        scene_idx = 0
        while scene_idx < n:
            start_f, end_f = bounds[scene_idx]
            if frame_idx < start_f:
                cap.grab()
                frame_idx += 1
                continue
            if frame_idx >= end_f:
                scene_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_m = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_m > best_sharpness[scene_idx]:
                best_sharpness[scene_idx] = blur_m
                best_frame[scene_idx] = frame.copy()

            frame_idx += 1

        # save only requested scenes
        for i in scene_indices:
            if best_frame[i] is not None:
                out = self.images_dir / f"scene_{i:03d}.jpg"
                cv2.imwrite(str(out), best_frame[i])
        cap.release()

    # ------------------------------------------------------------------
    # 2. Frame-difference dynamics
    # ------------------------------------------------------------------
    def _compute_dynamics(self):

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev = None
        diffs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = cv2.absdiff(gray, prev).mean()
                diffs.append(diff)
            prev = gray
        cap.release()

        # save whole vidoe plot
        plot_path = self.work_dir / f"video_dynamics.png"
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 2))
        plt.plot(diffs)
        plt.title(f"Video l1 dynamics")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        # build per-scene plots # advence
        with open(self.scenes_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["scene_idx"])
                start_f = int(row["start_frame"])
                end_f = int(row["end_frame"])
                segment = diffs[start_f:end_f]
                plot_path = self.plots_dir / f"scene_{idx:03d}_dynamics.png"
                if not plot_path.exists():
                    if not segment:
                        segment = [0.0]
                    self.scenes.append(
                        {
                            "idx": idx,
                            "start_time": float(row["start_time"]),
                            "end_time": float(row["end_time"]),
                            "start_frame": start_f,
                            "end_frame": end_f,
                            "avg_diff": float(np.mean(segment)),
                            "max_diff": float(np.max(segment)),
                        }
                    )
                    # save small plot
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(4, 2))
                    plt.plot(segment)
                    plt.title(f"Scene {idx} dynamics")
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()

    # ------------------------------------------------------------------
    # 3. Saliency
    # ------------------------------------------------------------------
    def _generate_saliency(self):
        """Motion-saliency for ALL frames → one mean heat-map per scene."""
        cap = cv2.VideoCapture(str(self.video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 1.  motion saliency init (once per movie)
        motion_sal = cv2.saliency.MotionSaliencyBinWangApr2014_create()
        motion_sal.setImagesize(w, h)
        if not motion_sal.init():
            raise RuntimeError("Motion saliency init failed")

        for sc in tqdm(self.scenes, desc="Saliency"):
            idx = sc["idx"]
            start_f = sc["start_frame"]
            end_f = sc["end_frame"]
            n_frames = end_f - start_f
            if n_frames <= 0:
                continue

            # accumulator for mean map
            mean_sal = np.zeros((h, w), dtype=np.float32)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            for _ in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, sal = motion_sal.computeSaliency(gray)  # sal: 0-1 float32
                if ok:
                    mean_sal += sal

            # if vw: vw.release()

            # ---- mean saliency → colour heat-map ----
            mean_sal /= n_frames
            mean_u8 = (mean_sal * 255).astype("uint8")
            heat = cv2.applyColorMap(mean_u8, cv2.COLORMAP_JET)

            # grab key-frame for pretty overlay
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ret, key = cap.read()
            if ret:
                final = cv2.addWeighted(key, 0.65, heat, 0.35, 0)
                out_file = self.saliency_dir / f"scene_{idx:03d}_mean_saliency.jpg"
                cv2.imwrite(str(out_file), final)

            # store raw mean map (float32) for later numeric scoring
            # npy_file = self.saliency_dir / f"scene_{idx:03d}_mean_saliency.npy"
            # np.save(npy_file, mean_sal)

        cap.release()

    # ------------------------------------------------------------------
    # 4. Gemini scoring
    # ------------------------------------------------------------------
    def _score_scenes(self):
        for sc in tqdm(self.scenes, desc="Gemini"):
            key_img = self.images_dir / f"scene_{sc['idx']:03d}.jpg"
            plot_img = self.plots_dir / f"scene_{sc['idx']:03d}_dynamics.png"
            description_path = (
                self.frame_that_reprezent_scene_descpriton_dir
                / f"scene_{sc['idx']:03d}.txt"
            )
            if not key_img.exists() or not plot_img.exists():
                continue
            if description_path.exists():
                json_str = description_path.read_text()
            else:
                json_str = self._gemini_score(key_img, plot_img)

                description_path.write_text(json_str)

            try:
                scores = json.loads(json_str)
                sc.update(scores)
            except Exception as e:
                print(f"[WARN] Gemini JSON parse fail scene {sc['idx']} : {e}")
                sc.update(
                    {
                        k: 0
                        for k in (
                            "PsychologicalHook",
                            "EmotionalTriggers",
                            "ContentsClarity",
                            "CompositionHierarchy",
                            "TextReadability",
                        )
                    }
                )

    def _gemini_score(self, key_img: Path, plot_img: Path) -> str:
        while True:
            try:
                model = genai.GenerativeModel(self.GEMINI_MODEL)
                with open(key_img, "rb") as f1, open(plot_img, "rb") as f2:
                    contents = [
                        self.VIDEO_SCORE_PROMPT,
                        {"mime_type": "image/jpeg", "data": f1.read()},
                        {"mime_type": "image/png", "data": f2.read()},
                    ]
                resp = model.generate_content(contents)
                text = resp.text.strip()
                if resp._done == True:
                    resp.usage_metadata
                    self.input_tokens[resp.model_version][
                        "input_tokens"
                    ] += resp.usage_metadata.prompt_token_count
                    self.output_tokens[resp.model_version][
                        "output_tokens"
                    ] += resp.usage_metadata.candidates_token_count
                resp = resp.text.strip()
                resp = resp.strip().removeprefix("```json").removesuffix("```").strip()
                return resp
            except Exception as e:
                print(f"[ERROR] Gemini fail: {e}  -> rotating key")
                self._rotate_key()
                time.sleep(5)

    def _global_whisper(self, audio_path: Path) -> Dict:
        """Return full whisper result with word-level timestamps."""
        print("[Audio] Running global whisper …")
        audio = whisper.load_audio(str(audio_path))
        model = whisper.load_model("tiny", device="cpu")
        return whisper.transcribe(model, audio, language="en")

    # ----------  new helper : slice & local whisper  ----------
    def _scene_audio_to_text(
        self, audio_path: Path, start: float, end: float, idx: int
    ) -> str:
        """Cut scene audio and re-transcribe locally."""
        duration = librosa.get_duration(path=str(audio_path))
        start = max(0.0, start)
        end = min(duration, max(start + 0.02, end))
        out_wav = self.scane_audio_dir / f"{idx}.wav"
        # exact cut with ffmpeg (fast & sample-accurate)
        (
            ffmpeg.input(str(audio_path), ss=start, t=end - start)
            .output(str(out_wav), loglevel="quiet", y=None)
            .run(overwrite_output=True)
        )
        audio = whisper.load_audio(str(out_wav))
        model = whisper.load_model("tiny", device="cpu")
        result = whisper.transcribe(model, audio, language="en")
        # return plain text
        return " ".join([s["text"].strip() for s in result["segments"]])

    # ----------  new helper : audio-quality metrics  ----------
    def _audio_metrics(self, audio_path: Path, start: float, end: float) -> Dict:
        """Compute SNR, RMS, peak, clipping, silence ratio."""
        y, sr = librosa.load(audio_path, offset=start, duration=end - start, sr=None)
        if y.size == 0:
            return {"snr": 0, "rms": 0, "peak": 0, "clip_ratio": 0, "silence_ratio": 0}

        # RMS & peak
        rms = float(librosa.feature.rms(y=y).mean())
        peak = float(np.abs(y).max())

        # clipping
        clip_ratio = float((np.abs(y) > 0.95).mean() * 100)

        # silence ratio (< -40 dB)
        silence = (librosa.amplitude_to_db(np.abs(y), ref=peak) < -40).mean()
        silence_ratio = float(silence * 100)

        # very simple SNR estimate (signal vs "quiet" parts)
        quiet = y[librosa.amplitude_to_db(np.abs(y), ref=peak) < -30]
        noise_power = np.mean(quiet**2) if quiet.size else 1e-8
        snr = 10 * np.log10((rms**2) / (noise_power + 1e-8))
        snr = max(float(snr), 0)

        return {
            "snr": snr,
            "rms": rms,
            "peak": peak,
            "clip_ratio": clip_ratio,
            "silence_ratio": silence_ratio,
        }

    # ----------  new helper : LLM visual-vs-audio match score  ----------
    def _match_score(self, audio_path: Path, image_path: Path) -> int:
        while True:
            try:
                model = genai.GenerativeModel(self.GEMINI_MODEL)
                with open(image_path, "rb") as f1, open(audio_path, "rb") as f2:
                    resp = model.generate_content(
                        [
                            self.MEDIA_SCORE_PROMPT,
                            {"mime_type": "image/jpeg", "data": f1.read()},
                            {"mime_type": "audio/wav", "data": f2.read()},
                        ]
                    )
                if resp._done == True:
                    resp.usage_metadata
                    self.input_tokens[resp.model_version][
                        "input_tokens"
                    ] += resp.usage_metadata.prompt_token_count
                    self.output_tokens[resp.model_version][
                        "output_tokens"
                    ] += resp.usage_metadata.candidates_token_count
                resp = resp.text.strip()
                resp = resp.strip().removeprefix("```json").removesuffix("```").strip()
                return resp
            except Exception as e:
                print(f"[WARN] match score fail: {e}  -> rotate key")
                self._rotate_key()
                time.sleep(5)

    # ----------  integrate into existing pipeline  ----------
    def _audio_pipeline(self):
        """Master audio step: run once after scene detection."""
        print("[Audio] Demuxing audio track …")
        # 1. extract whole audio
        whole_audio = self.work_dir / "full_audio.wav"
        (
            ffmpeg.input(str(self.video_path))
            .output(
                str(whole_audio), acodec="pcm_s16le", ac=1, ar="16000", loglevel="quiet"
            )
            .run(overwrite_output=True)
        )

        # 2. global whisper (for reference & boundary check)

        whole_transcript_audio_path = self.work_dir / "full_audio_transcript.json"
        if not whole_transcript_audio_path.exists():
            global_result = self._global_whisper(whole_audio)
            with open(whole_transcript_audio_path, "w", encoding="utf-8") as f:
                json.dump(global_result, f, indent=4, ensure_ascii=False)

        for sc in tqdm(self.scenes, desc="Audio"):
            audio_transcript_path = self.scane_audio_transcript_dir / f"{idx}.txt"
            if audio_transcript_path.exists():
                continue
            idx = sc["idx"]
            start = sc["start_time"]
            end = sc["end_time"]
            text = self._scene_audio_to_text(whole_audio, start, end, idx)
            audio_transcript_path.write_text(text)

        for sc in tqdm(self.scenes, desc="Audio"):
            idx = sc["idx"]
            start = sc["start_time"]
            end = sc["end_time"]

            audio_metric = self._audio_metrics(whole_audio, start, end)
            audio_metric_path = self.scane_audio_metric_dir / f"{idx}.json"
            with open(audio_metric_path, "w", encoding="utf-8") as f:
                json.dump(audio_metric, f, indent=4, ensure_ascii=False)

            # audio quality
            sc.update(audio_metric)

        # 4. neighbour loudness mismatch
        for prev, nxt in zip(self.scenes, self.scenes[1:]):
            delta_db = 20 * np.log10(nxt["rms"] / (prev["rms"] + 1e-8))
            prev["loudness_change_next_db"] = float(delta_db)

    def _llm_description_of_scene(self):
        for sc in tqdm(self.scenes, desc="Audio"):
            idx = sc["idx"]
            start = sc["start_time"]
            end = sc["end_time"]
            # visual-audio match
            key_img = self.images_dir / f"scene_{idx:03d}.jpg"
            heat_map_img = self.saliency_dir / f"scene_{idx:03d}_mean_saliency.jpg"
            audio_wav = self.scane_audio_dir / f"{idx}.wav"
            visual_audio_match_file_path = self.visual_audio_match_path / f"{idx}.json"

            if not visual_audio_match_file_path.exists():
                result = self._match_score(audio_wav, key_img)
                result = json.loads(result)
                new_result = {}
                for k, v in result.items():
                    if v is None:
                        v = 0
                    new_result[k] = v
                result = new_result
                with open(visual_audio_match_file_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                sc.update(result)
            else:
                with open(visual_audio_match_file_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                new_result = {}
                for k, v in result.items():
                    if v is None:
                        v = 0
                    new_result[k] = v
                result = new_result
                sc.update(result)

    # ------------------------------------------------------------------
    # 5. Meta-analysis
    # ------------------------------------------------------------------
    def _build_metadata(self):
        durations = [s["end_time"] - s["start_time"] for s in self.scenes]

        transcript_path = self.work_dir / "full_audio_transcript.json"
        with open(transcript_path, "r", encoding="utf-8") as f:
            tanscript = json.load(f)

        self.meta = {
            "num_scenes": len(self.scenes),
            "total_duration": sum(durations),
            "avg_scene_duration": float(np.mean(durations)),
            "std_scene_duration": float(np.std(durations)),
            "tanscript": tanscript["text"],
        }

        # scenes llm part
        llm_keys = [
            "video_viewer_engagement",
            "video_emotional_impact",
            "video_content_clarity",
            "video_visual_hierarchy",
            "video_text_readability",
            "video_retention_score",
            "audio_voice_clarity",
            "audio_emotional_impact",
            "audio_content_relevance",
            "audio_technical_quality",
            "audio_visual_match",
            "topic_sensitivity_score",
        ]

        for key in llm_keys:
            values = [
                (s[key], s)
                for s in self.scenes
                if key in s and isinstance(s[key], (int, float))
            ]

            if values:
                vals, scenes = zip(*values)
                min_val, max_val = min(vals), max(vals)
                min_sc, max_sc = (
                    scenes[vals.index(min_val)],
                    scenes[vals.index(max_val)],
                )

                self.meta[f"{key}_min"] = min_val
                self.meta[f"{key}_max"] = max_val
                self.meta[f"{key}_mean"] = float(np.mean(vals))
                self.meta[f"{key}_idx_min"] = int(min_sc["idx"])
                self.meta[f"{key}_idx_max"] = int(max_sc["idx"])
                self.meta[f"{key}_start_min"] = float(min_sc["start_time"])
                self.meta[f"{key}_end_min"] = float(min_sc["end_time"])
                self.meta[f"{key}_start_max"] = float(max_sc["start_time"])
                self.meta[f"{key}_end_max"] = float(max_sc["end_time"])

        metadata_path = self.work_dir / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=4, ensure_ascii=False)

        scenes_metadata_dir_path = self.work_dir / "scenes_metadata"
        scenes_metadata_dir_path.mkdir(exist_ok=True, parents=True)

        for scene_idx, scene in enumerate(self.scenes):
            scene_file_path = scenes_metadata_dir_path / f"{scene_idx}.json"
            with open(scene_file_path, "w", encoding="utf-8") as f:
                json.dump(scene, f, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6. Final report
    # ------------------------------------------------------------------
    def _write_report(self) -> Path:
        payload = {
            "meta": self.meta,
        }
        final_request_payload_file_path = self.work_dir / "final_request_payload.md"
        final_request_payload_file_path.write_text(json.dumps(payload), encoding="utf8")
        while True:
            try:
                model = genai.GenerativeModel(self.FINAL_REPORT_MODEL)
                resp = model.generate_content(
                    [self.FINAL_REPORT_PROMPT, json.dumps(payload)]
                )
                markdown = resp.text.strip()
                if resp._done == True:
                    resp.usage_metadata
                    self.input_tokens[resp.model_version][
                        "input_tokens"
                    ] += resp.usage_metadata.prompt_token_count
                    self.output_tokens[resp.model_version][
                        "output_tokens"
                    ] += resp.usage_metadata.candidates_token_count
                break
            except Exception as e:
                print(f"[ERROR] Final report fail: {e}  -> rotating key")
                self._rotate_key()
                time.sleep(5)
        report_path = self.work_dir / "report.md"
        report_path.write_text(markdown, encoding="utf8")
        return report_path

    def save_costs(self):
        total_cost = 0.0
        for model_name, model_input_tokens_number in self.input_tokens.items():
            total_cost += (
                gemini_pricing[model_name]["input_tokens_cost"]
                * model_input_tokens_number["input_tokens"]
            )

        for model_name, model_output_tokens_number in self.output_tokens.items():
            total_cost += (
                gemini_pricing[model_name]["output_tokens_cost"]
                * model_output_tokens_number["output_tokens"]
            )

        price_raport = {"total_cost": total_cost}
        print(price_raport)
        result = json.loads(price_raport)
        price_raport_path = self.work_dir / "costs.json"
        new_result = {}
        for k, v in result.items():
            if v is None:
                v = 0
            new_result[k] = v
        result = new_result
        with open(price_raport_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _configure_api(self):
        genai.configure(api_key=self._api_keys[self._key_idx])

    def _rotate_key(self):
        self._key_idx = (self._key_idx + 1) % len(self._api_keys)
        self._configure_api()

    # optional cleanup
    def cleanup(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)


if __name__ == "__main__":
    # video_file = Path(__file__).parent.parent / "media" / "raw"/ "Apple’s New AI SHOCKS The Industry With 85X More Speed (Beating Everyone).mp4"
    # video_file = Path(__file__).parent.parent / "output" / "Embodied-R1_Reinforced_Embodied_Reasoning_for_General_Robotic_Manipulation.mp4"
    video_file = (
        Path(__file__).parent
        / "40k Daemons outside the Voidship window during warp travel.mp4"
    )
    output_dir = Path(__file__).parent / "analysis_output2" / video_file.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    inspector = VideoInspector(video_file, work_dir=output_dir)
    report = inspector.analyse()
    print(report)
