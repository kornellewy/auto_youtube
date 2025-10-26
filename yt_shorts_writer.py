"""
load ai paper and all descritpion of infografic 
"""

from __future__ import annotations
from pathlib import Path
from html2image import Html2Image
import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
#  CONFIG – change only these variables
MODEL_NAME = "gemini-2.5-flash"  # "gemini-2.5-flash"
BACKAP_MODEL_NAME = "gemini-2.5-flash"
GEMINI_KEY   = genai.configure(api_key=os.getenv("GOOGLE_API_KEY4"))


SHORTS_PROMPT = """
You are a viral YouTube-Shorts script writer.
I will give you:
1. The full paper text.
2. A list of the infographics descriptions we produced.
3. A list of the figures descriptions.

Your job:
- Create between 5 and 10 ultra-dense 15–30-second scripts.
- Each script MUST literally contain the exact phrase “Infographic X” or “Figure X” (with the correct number) once – this is how we sync voice-over to the visual.
- Follow the BEST-practice checklist below for every script.

BEST PRACTICE CHECKLIST
1. HOOK (first 1-3 s): punchy, curiosity-driven, capitalist / money / efficiency angle.
2. BODY: 1-3 lightning facts taken straight from the paper; use numbers & superlatives.
3. CTA (last 2-3 s): “Watch the full breakdown on … link below!”
4. Style: second-person, hype, simple words, no jargon.
5. Length: 45-55 words ≈ 15-20 s when read fast.

Return the scripts as plain text blocks separated only by two newlines:

===== START TAKEAWAY 1 =====
Script 1
===== END TAKEAWAY 1 =====

(etc.)

Input

Paper Source: {article}

Image Descriptions: {images}

Infographics: {infographics}

"""

def create_shorts_scripts(paper_path: Path) -> list[str]:
    have_short_script = False
    for txt_path in paper_path.glob("*.txt"):
        if "yt_short_script_" in txt_path.stem:
            have_short_script = True
            break
    if have_short_script:
        return

    # load article
    article_text = (Path(paper_path) / "article.txt").read_text(encoding="utf8")
    # load all infografic
    infografic_descriptions = []
    all_txt = list(paper_path.rglob("*.txt"))
    all_infografic_txt = [txt_path
        for txt_path in all_txt
        if txt_path.stem.startswith("infografic_")]
    for infografic_txt in all_infografic_txt:
        image_path = infografic_txt.with_suffix(".jpg")
        if infografic_txt.exists() and  image_path.exists():
            description = infografic_txt.read_text(encoding="utf-8")
            infografic_descriptions.append(f"{image_path.name}: {description.strip()}")
    infografic_descriptions = "\n".join(infografic_descriptions)
    # load all figures
    image_desc = []
    for image_path in sorted(paper_path.glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists() and "image_" in image_path.stem:
                description = txt_path.read_text(encoding="utf-8")
                image_desc.append(f"{image_path.name}: {description.strip()}")
    image_desc =  "\n".join(image_desc)

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY3"))
    model = genai.GenerativeModel(MODEL_NAME)
    full_prompt = SHORTS_PROMPT.format(article=article_text, images=image_desc, infographics=infografic_descriptions)
    reply = model.generate_content(full_prompt)

    blocks = reply.text.split("===== START TAKEAWAY")[1:]

    scripts_paths = []
    for b in blocks:
        short_text = b.split("===== END TAKEAWAY")[0].strip()
        num = short_text.split("=====")[0].strip()
        script = short_text.split("=====\n")[1]
        yt_short_script_path = paper_path / f"yt_short_script_{num}.txt"
        yt_short_script_path.write_text(script, encoding="utf8")
        scripts_paths.append(yt_short_script_path)
    return scripts_paths

if __name__ == "__main__":
    create_shorts_scripts(Path("/media/kornellewy/jan_dysk_3/auto_youtube/scraped_articles/EvoTest_Evolutionary_Test-Time_Learning_for_Self-Improving_Agentic_Systems"))