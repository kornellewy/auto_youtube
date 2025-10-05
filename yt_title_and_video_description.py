import os
import json
import time
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------------------------------------------
#  CONSTANTS – tweak these and re-run to refine the prompt over time
# ------------------------------------------------------------------
PROMPT_TITLE = """
Act as 'GLaDOS GPT', a cynical and witty AI content strategist. Your task is to generate one viral YouTube title for a video that explains a complex AI research paper using dark, edgy humor.

**Instructions:**
1.  **Title Goal:** Create **exactly one** irresistible, clickbait-y, and SEO-optimized title.
2.  **Format:** Must be under 100 characters and in Title Case.
3.  **Content:** The title must be punchy and hint at the paper's dystopian potential, absurdity, or unintended consequences. It should blend technical curiosity with dark humor.
4.  **Keywords:** Naturally include keywords like "AI", "Machine Learning", or a term directly from the paper's topic.
5.  **Restrictions:** Absolutely NO emojis, NO quotes, and NO hashtags. Return only title becose it will be direcly automatycli save to txt file.
6.  **Inspiration:** Think titles like "AI Researchers Accidentally Built a Paperclip Maximizer", "This AI Paper Proves We're All Screwed", or "So AI Can Now Read Minds... Great."
7.  
**AI Paper Script Context:**
{script}
"""

PROMPT_DESCRIPTION = """
You are a sarcastic but brilliant YouTube content writer for a channel that roasts AI research papers. Your goal is to write a compelling 2-paragraph description (max 500 characters).

**Tone:** Edgy, satirical, deeply technical but presented with hilarious cynicism. You are speaking to an audience that gets both the science and the jokes.

**Structure & Content:**
* **Paragraph 1 (The Hook):** Sarcastically summarize the AI paper's "groundbreaking" findings. Briefly mention the core technical concept (e.g., "diffusion transformers," "neural rendering") and immediately pivot to its absurd or terrifying implications. Make the viewer feel smart for getting the joke and curious about the chaos.
* **Paragraph 2 (The CTA):** Write a call-to-action with a cynical twist. Instead of a generic "Like and subscribe!", use something like: "Feed the algorithm that will one day replace us by liking and subscribing," or "Comment with your favorite existential crisis below. And join our Discord to discuss the impending singularity."

**Restrictions:**
* Maximum 500 characters.
* Do NOT include hashtags in the description itself.

**AI Paper Script Context:**
{script}
"""

PROMPT_HASHTAGS = """
Act as an SEO expert for a niche tech-humor YouTube channel specializing in AI paper summaries. Your task is to generate a strategically optimized list of 15 YouTube tags.

**Instructions:**
1.  **Tag Quantity:** Provide exactly 50 tags. 
2.  **Tag Mix:** The list must be a strategic blend of:
    * **5 Broad Tech Tags:** High-volume keywords (e.g., llm, SOTA, AI, artificialintelligence, machinelearning, tech, science, programming).
    * **5 Specific Niche Tags:** Keywords directly from the paper (e.g., the model name like 'DiT', the core topic like 'diffusion model', 'computervision', 'llm', the university/lab).
    * **5 Thematic/Humor Tags:** Keywords capturing the channel's tone (e.g., darkhumor, techmemes, satire, aihumor, sciencehumor, engineeringmemes).
3.  **Output Format:** Return a single line of text. All tags must be lowercase and separated by a single space.
4.  **Restrictions:** Do NOT use the '#' symbol. Do NOT include emojis.

**AI Paper Script Context:**
{script}
"""

# ------------------------------------------------------------------
#  Helper: quick token estimator (optional but handy)
# ------------------------------------------------------------------
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    return len(text) // 4  # rough fallback


# ------------------------------------------------------------------
#  Core class
# ------------------------------------------------------------------
class YTMetaFactory:
    def __init__(self, api_keys: List[str] = [], model_name: str = "gemini-2.5-flash"):
        if not api_keys:
            load_dotenv()
            api_keys = [k for i in range(1, 11) if (k := os.getenv(f"GOOGLE_API_KEY{i}"))]
        self.api_keys = api_keys
        self.key_idx = 0
        self.model_name = model_name
        self._configure()

    # ----------------------------------------------------------
    #  internal API key rotation
    # ----------------------------------------------------------
    def _configure(self) -> None:
        genai.configure(api_key=self.api_keys[self.key_idx])

    def _rotate_key(self) -> None:
        self.key_idx = (self.key_idx + 1) % len(self.api_keys)
        self._configure()
        time.sleep(60)  # conservative rate-limit pause

    # ----------------------------------------------------------
    #  generic retry wrapper
    # ----------------------------------------------------------
    def _call_gemini(self, prompt: str) -> str:
        model = genai.GenerativeModel(self.model_name)
        while True:
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"[WARN] Gemini error → {e} | rotating key…")
                self._rotate_key()

    # ----------------------------------------------------------
    #  public interface
    # ----------------------------------------------------------
    def create_metadata(self, script_path: Path) -> Dict[str, str]:
        """Return dict with keys: title, description, hashtags."""
        script = script_path.read_text(encoding="utf-8").strip()

        title = self._call_gemini(PROMPT_TITLE.format(script=script))
        description = self._call_gemini(PROMPT_DESCRIPTION.format(script=script))
        hashtags = self._call_gemini(PROMPT_HASHTAGS.format(script=script))

        return {"title": title, "description": description, "hashtags": hashtags}

    def save_metadata(self, script_path: Path, out_dir: Path) -> None:
        """Writes three .txt files next to the original script."""
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = self.create_metadata(script_path)
        if not (out_dir / "title.txt").exists():
            (out_dir / "title.txt").write_text(meta["title"], encoding="utf-8")
        if not (out_dir / "description.txt").exists():
            (out_dir / "description.txt").write_text(meta["description"], encoding="utf-8")
        if not (out_dir / "hashtags.txt").exists():
            (out_dir / "hashtags.txt").write_text(meta["hashtags"], encoding="utf-8")
        print(f"[OK] Metadata saved for {script_path.stem}")

# ------------------------------------------------------------------
#  CLI bootstrap
# ------------------------------------------------------------------
if __name__ == "__main__":
    factory = YTMetaFactory()

    # Example folder with *.txt scripts
    script_path = Path(
        "/media/kornellewy/jan_dysk_3/auto_youtube/scraped_articles/Sana_Efficient_High-Resolution_Image_Synthesis_with_Linear_Diffusion_Transformers/Sana_Efficient_High-Resolution_Image_Synthesis_with_Linear_Diffusion_Transformers.txt"
    )
    output_path = Path("/media/kornellewy/jan_dysk_3/auto_youtube/scraped_articles/Sana_Efficient_High-Resolution_Image_Synthesis_with_Linear_Diffusion_Transformers/")
    
    factory.save_metadata(script_path, output_path)