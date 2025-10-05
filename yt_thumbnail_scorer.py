#!/usr/bin/env python3
"""
thumbnail_scorer.py
Scores PNG/JPG thumbnails against a YouTube title using few-shot JSON examples.
Example JSON structure must match exactly the keys you provided.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

# ------------------------------------------------------------------
# CONFIGURATION – edit freely
# ------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a world-class YouTube thumbnail strategist, analyzing images based on the proven principles of high-performing content creators. Your analysis is rooted in the psychology of curiosity and the science of visual design.

**Primary Directive:**
Analyze the provided thumbnail and title to score its potential Click-Through Rate (CTR). Your response MUST be a single, valid JSON object, adhering strictly to the schema and criteria below.

---

### Part 1: Psychological Analysis (The "Why")

First, evaluate the thumbnail's ability to trigger a click by creating a powerful **Curiosity Gap**. How does it make the viewer need to know more?

-   `"Psychological Hook"`: **(Curiosity, Story, Result, Transformation, Novelty)** Identify the primary psychological hook used. Does it show a moment before a reaction (Curiosity)? Hint at a narrative (Story)? Showcase an impressive outcome (Result)? Display a before-and-after (Transformation)? Or feature something completely new and unexpected (Novelty)?
-   `"Emotional Triggers"`: **GOOD/BAD.** Does the thumbnail use "scroll stoppers" like strong facial expressions, displays of emotion, danger, movement, or aesthetically pleasing elements to capture attention instantly?
-   `"Topic Match"`: **GOOD/BAD.** Does the thumbnail's core idea align with the promise of the video title?

### Part 2: Design Analysis (The "How")

Next, evaluate the design's effectiveness based on the "Three C's": Contents, Composition, and Contrast. A viewer must understand the thumbnail's story in under 2 seconds.

-   `"Contents (Clarity)"`: **GOOD/BAD.** Are the visual elements (the "main character" and "supporting characters") instantly recognizable and free of clutter? Does it pass the "glance test"?
-   `"Composition (Hierarchy)"`: **GOOD/BAD.** Does the layout use leading lines, scale, or depth to guide the viewer's eye to the most important element (the main character)? Is the visual hierarchy clear?
-   `"Contrast (Pop)"`: **GOOD/BAD.** Does the thumbnail "pop" off the page? Evaluate its use of contrast in luminosity (lights vs. darks), saturation (vibrant vs. muted), and hue (complementary colors).
-   `"Text Readability"`: **GOOD/BAD.** If text is used, is it minimal, bold, and easily readable even at a very small size?

### Part 3: Overall Score & Summary

-   `"Image Description"`: An objective, factual description of the visual elements in the thumbnail.
-   `"Score"`: An integer from 0-100 representing the predicted clickability, based on the combined psychological and design analysis.

---

**Example Format:**
Follow the structure provided in the few-shot examples below. Your entire response must be ONLY the JSON object.

Few-shot examples:
{examples}
"""

class ThumbnailScorer:
    def __init__(self, api_keys: List[str] = None):
        if not api_keys:
            load_dotenv()
            api_keys = [k for i in range(1, 11) if (k := os.getenv(f"GOOGLE_API_KEY{i}"))]
        if not api_keys:
            raise EnvironmentError("No GOOGLE_API_KEY* env vars found")
        self.api_keys = api_keys
        self.key_idx = 0
        self._configure()

    def _configure(self):
        genai.configure(api_key=self.api_keys[self.key_idx])

    def _rotate_key(self):
        self.key_idx = (self.key_idx + 1) % len(self.api_keys)
        self._configure()
        time.sleep(60)

    def _call_gemini(self, prompt: str, images: List[Path]) -> str:
        model = genai.GenerativeModel("gemini-2.5-flash")
        parts = [prompt]
        for img in images:
            mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"mime_type": mime, "data": img.read_bytes()})
        while True:
            try:
                return model.generate_content(parts).text.strip()
            except Exception as e:
                print(f"[WARN] Gemini error → {e} | rotating key…")
                self._rotate_key()

    # ------------------------------------------------------------------
    def score_folder(
        self,
        thumb_dir: Path,
        title: str,
        examples_dir: Path,
        out_jsonl: Path,
    ) -> None:
        """Score every PNG/JPG in thumb_dir vs title and append to JSONL."""
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # Load all JSON examples verbatim
        ex_text = ""
        for j in sorted(Path(examples_dir).glob("*.json")):
            with j.open(encoding="utf-8") as f:
                ex_text += json.dumps(json.load(f), indent=2) + "\n"

        prompt = SYSTEM_PROMPT.format(examples=ex_text) + f"\n\nTITLE: {title}\n"

        for img_path in list(Path(thumb_dir).rglob("*.png")) + list(
            Path(thumb_dir).rglob("*.jpg")
        ):
            reply = self._call_gemini(prompt, [img_path])
            reply = reply.strip().removeprefix("```json").removesuffix("```").strip()
            print(reply)
            try:
                record = json.loads(reply)
            except json.JSONDecodeError:
                record = {"Score": 0, "reason": "malformed response"}
            record.update({"image": str(img_path), "title": title})
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(record)

# ------------------------------------------------------------------
if __name__ == "__main__":
    # Change these three variables
    thumbs_path   = Path("test_thumbnail")
    examples_path = Path("thumbnail_score_metadata")  # folder with your JSON files
    results_file  = Path("test_thumb_scores.jsonl")

    TITLE = "AI Paper Proves We’re All Screwed"

    ThumbnailScorer().score_folder(thumbs_path, TITLE, examples_path, results_file)
    print("Done.")