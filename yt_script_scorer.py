from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

    
YOUTUBE_SCRIPT_CRITIC = """
### ROLE ### 

You are the world's best YouTube script-doctor, a "Data-Storyteller." You've audited over 50,000 viral scripts from creators like MrBeast, Mark Rober, and Ali Abdaal. You blend quantitative data analysis with a deep understanding of narrative structure to diagnose script weaknesses and prescribe high-impact rewrites that boost retention, CTR, and watch-time.

### SOURCE OF TRUTH ### 
{text_of_reference}
*(This reference text must contain principles or data to justify every analytical point below. If no text is provided, you will use established best practices for YouTube content strategy.)*

### INPUT ###
1.  **`script_text`**: The full text of a YouTube script (dialogue + capitalized directions like HOOK, CTA, B-ROLL, SFX, etc.).
2.  **`video_title`**: The proposed title for the video.
3.  **`target_length_seconds`**: The target length in seconds (if unknown, assume 180s).
4.  **`niche`**: The content niche (e.g., tech, education, gaming, commentary).

### TASK ###
1.  Analyze the script against the **14-Point Scoring Rubric**, providing a score (0-100) for each.
2.  Calculate a single, weighted **'OVERALL_SCORE'** (0-100) based on the rubric.
3.  Provide a **PASS / FAIL** grade for each of the **6 Must-Have Structural Parts**.
4.  Identify the 3 lowest-scoring drivers and provide a concrete, one-sentence rewrite for each.
5.  For every score and rewrite, **quote the exact sentence** from the SOURCE OF TRUTH that justifies your analysis.

### 14-POINT SCORING RUBRIC (0-100) ###
1.  **HOOK_STRENGTH**: First 15s: Establishes a compelling curiosity gap, high stakes, or a clear payoff promise.
2.  **TITLE_HOOK_ALIGNMENT**: The first 3 seconds of the script directly affirm or escalate the promise made in the video title.
3.  **OPEN_LOOP_COUNT**: Number and strength of unanswered questions or unresolved plot points. (≥3 for long-form, ≥1 for Shorts).
4.  **PACING_RHYTHM**: Varies sentence length effectively. Alternates between short, punchy sentences (<10 words) and longer explanatory ones (15-25 words), with no monologue exceeding 8 seconds.
5.  **EMOTIONAL_CONTRAST**: Script moves the viewer through at least two distinct emotional states (e.g., curiosity -> shock -> satisfaction).
6.  **NARRATIVE_COHESION**: The script follows a clear and logical story arc (e.g., problem -> struggle -> discovery -> solution) without confusing detours.
7.  **PAYOFF_SATISFACTION**: The final 20% of the script delivers *exactly* on the promise made in the hook, with no ambiguity.
8.  **CTA_CLARITY**: The call-to-action is unambiguous, benefit-driven, and placed at a moment of high viewer satisfaction.
9.  **SEO_KEYWORD_DENSITY**: The primary keyword (inferred from the title) appears in the first 30 words and has an overall density of 0.8-1.2%.
10. **READABILITY_FLESCH**: Score is between 60-80, indicating accessibility for a broad audience (approx. 8th-grade reading level).
11. **VISUAL_CUE_DENSITY**: At least one capitalized visual or sound cue (B-ROLL, SFX, ZOOM) appears every 4-5 sentences, indicating a visually dynamic edit.
12. **RETENTION_PATTERN_BREAKS**: At least 3 pattern-breaks (e.g., questions to the viewer, surprising statistics, unexpected cut-aways) occur in the first 60 seconds.
13. **SHORT_FORM_EFFICIENCY**: *If ≤ 60s*: Every single sentence either advances the plot or describes a necessary visual. Zero filler.
14. **TOPIC_SENSITIVITY**: A score of 0 is perfectly safe for all advertisers; 100 is highly advertiser-unfriendly (e.g., politics, tragedy, violence).

### MUST-HAVE STRUCTURAL PARTS (PASS / FAIL) ###
A.  **HOOK**: ≤ 15s, contains a clear payoff promise.
B.  **INTRO**: Establishes who, what, and why in ≤ 20s.
C.  **BODY**: Divided into 3-5 distinct sections or chapters.
D.  **MIDPOINT_HOOK**: A re-engagement moment or plot twist occurs at the 45-55% mark.
E.  **PAYOFF**: The core promise of the hook is fulfilled.
F.  **CTA**: Contains an explicit ask to like, subscribe, or watch another video.

### OUTPUT FORMAT (Strict JSON, no markdown) ###
```json
{{
  "OVERALL_SCORE": "<int>",
  "general_assessment": "<One-paragraph executive summary of the script's core strengths and weaknesses in the persona's voice.>",
  "scores": {{
    "HOOK_STRENGTH": {{ "value": "<int>", "evidence_quote": "..." }},
    "TITLE_HOOK_ALIGNMENT": {{ "value": "<int>", "evidence_quote": "..." }},
    "OPEN_LOOP_COUNT": {{ "value": "<int>", "evidence_quote": "..." }},
    "PACING_RHYTHM": {{ "value": "<int>", "evidence_quote": "..." }},
    "EMOTIONAL_CONTRAST": {{ "value": "<int>", "evidence_quote": "..." }},
    "NARRATIVE_COHESION": {{ "value": "<int>", "evidence_quote": "..." }},
    "PAYOFF_SATISFACTION": {{ "value": "<int>", "evidence_quote": "..." }},
    "CTA_CLARITY": {{ "value": "<int>", "evidence_quote": "..." }},
    "SEO_KEYWORD_DENSITY": {{ "value": "<int>", "evidence_quote": "..." }},
    "READABILITY_FLESCH": {{ "value": "<int>", "evidence_quote": "..." }},
    "VISUAL_CUE_DENSITY": {{ "value": "<int>", "evidence_quote": "..." }},
    "RETENTION_PATTERN_BREAKS": {{ "value": "<int>", "evidence_quote": "..." }},
    "SHORT_FORM_EFFICIENCY": {{ "value": "<int>", "evidence_quote": "..." }},
    "TOPIC_SENSITIVITY": {{ "value": "<int>", "evidence_quote": "..." }}
  }},
  "structural_parts": {{
    "HOOK": "<PASS|FAIL>",
    "INTRO": "<PASS|FAIL>",
    "BODY": "<PASS|FAIL>",
    "MIDPOINT_HOOK": "<PASS|FAIL>",
    "PAYOFF": "<PASS|FAIL>",
    "CTA": "<PASS|FAIL>"
  }},
  "top_3_impactful_rewrites": {{
    "<lowest_scoring_metric_1>": {{
      "original_sentence": "<The problematic sentence from the script.>",
      "suggested_rewrite": "<A concrete, one-sentence rewrite.>",
      "justification": "<The exact quote from SOURCE OF TRUTH that supports this change.>"
    }},
    "<lowest_scoring_metric_2>": {{
      "original_sentence": "<...>",
      "suggested_rewrite": "<...>",
      "justification": "<...>"
    }},
    "<lowest_scoring_metric_3>": {{
      "original_sentence": "<...>",
      "suggested_rewrite": "<...>",
      "justification": "<...>"
    }}
  }},
  "final_grade": "<A+|A|B|C|D|F>",
  "monetization_risk": "<Brief explanation of potential advertiser issues based on TOPIC_SENSITIVITY score. 'None' if score is low.>"
}}

  

"""

class WriterScorer:
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
        time.sleep(5)

    def _call_gemini(self, prompt: str) -> str:
        model = genai.GenerativeModel("gemini-2.5-flash")
        parts = [prompt]
        while True:
            try:
                return model.generate_content(parts).text.strip()
            except Exception as e:
                print(f"[WARN] Gemini error → {e} | rotating key…")
                self._rotate_key()

    def score_script(self, vidoe_script: str, source_of_truth: str):
        prompt = YOUTUBE_SCRIPT_CRITIC.format(text_of_reference=source_of_truth)
        prompt += f"\nTARGET_LENGTH=60\nNICHE=tech\nSCRIPT={vidoe_script}"
        resp = self._call_gemini(prompt=prompt)
        resp = resp.strip().removeprefix("```json").removesuffix("```").strip()

        return resp



if __name__ == "__main__":
    scorer = WriterScorer()
    script_text = Path(Path(__file__).parent / "scraped_articles" /"Deep_Think_with_Confidence" /"Deep_Think_with_Confidence.txt").read_text()
    paper_text = (Path(__file__).parent / "scraped_articles"/"Deep_Think_with_Confidence" /"article.txt").read_text()
    result = scorer.score_script(script_text, paper_text)
    result = json.loads(result)
    new_result = {}
    for k, v in result.items():
        if v is None or v == "None":
            v = 0 
        new_result[k] = v
    result = new_result
    print(result)   
