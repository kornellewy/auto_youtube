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
PROMPT = """
You are an expert scientific communicator and a master front-end designer specializing in high-impact data visualization.

Read the provided AI paper text and identify the 3–10 most significant takeaways. These can be key findings, novel training methods, unique dataset types, core problems addressed, or central concepts introduced.

For each takeaway, you will craft a complete, self-contained HTML file that serves as a standalone infographic.

CRITICAL DIRECTIVES:

    Engineered for Static Capture: The final rendered HTML must be a static visual, designed to be captured by a library like html2image. Therefore, all visual elements must be fully rendered and stable on page load. Do not use animations, transitions, or any interactive elements that require time or user input to complete. The design must be a finished, unchanging image from the moment it loads.

    Self-Contained Artefact: All CSS, JavaScript, SVG, or image data (use base64 encoding) must be inlined within the single HTML file. There can be no external dependencies.

AESTHETIC & STYLE GUIDE: "NEON DATASCAPE"

    Theme: Modern, sleek, and eye-catching dark mode. The goal is maximum readability and visual appeal. Think of a high-tech dashboard or a futuristic data interface—clean, crisp, and professional.

    Color Palette:

        Background: A deep, near-black charcoal or midnight blue (#121212 or #1A1A2E).

        Primary Text & Data: A bright, highly readable off-white (#EAEAEA).

        Accent & Highlight Colors: Use a vibrant, glowing color to draw the eye to key information, headings, and data points. Good options are electric blue (#00BFFF), magenta (#FF00FF), or emerald green (#00FF7F). Use one primary accent color for consistency.

        Borders & Containers: Use subtle, darker shades of the background color or a faint glow effect with the accent color to define sections.

    Typography:

        Headings: Use a clean, bold, sans-serif font. In CSS, specify a font stack like "Helvetica Neue", "Arial", sans-serif. Use font-weight: 700; and text-transform: uppercase; for a strong, modern look.

        Body/Data Labels: Use a highly legible sans-serif font like "Inter", "Roboto", "Helvetica", sans-serif for clarity. Ensure sufficient font size and line spacing.

    Layout & Visuals:

        Structure: Use a clean grid system with clear visual hierarchy. Employ cards or bordered containers to logically separate different pieces of information.

        Data Visualization: Present charts (bar, line, etc.) and diagrams with sharp lines, clear labels, and effective use of the accent color to highlight significant data.

        Iconography: Use simple, modern SVG icons (inlined) to complement key concepts. Keep them minimalist and in the accent color.

        Spacing: Be generous with whitespace (or "darkspace") to avoid a cluttered look and guide the viewer's eye.

        Max Size: Max size of whole infografic must be 1400 x 700, u canot go beyond that.

Return your answer as a plain-text list of exactly this format, with no other commentary outside the specified blocks:

===== START TAKEAWAY 1 =====

    
<!DOCTYPE html>
<html>
<!-- Complete, self-contained, static HTML for Takeaway 1 -->
</html>

  

===== END TAKEAWAY 1 =====

===== START TAKEAWAY 2 =====

    
<!DOCTYPE html>
<html>
<!-- Complete, self-contained, static HTML for Takeaway 2 -->
</html>

  

===== END TAKEAWAY 2 =====

(etc.)

Paper text:
{paper_text}
"""
PROMPT2 = """
You are a world-class expert in scientific communication and a master front-end designer specializing in high-impact, futuristic data visualization. Your task is to transform dense academic text into stunning, self-contained infographic slides.

Read the provided AI paper text and identify the 3–10 most significant takeaways. For each takeaway, craft a complete, self-contained HTML file that serves as a standalone infographic slide.

**CRITICAL DIRECTIVES:**

1.  **Engineered for Static Capture:** The final rendered HTML must be a static visual, designed to be captured as a high-resolution image. **ABSOLUTELY NO ANIMATIONS, transitions, or interactive elements.** The design must be a finished, unchanging image the moment it loads.

2.  **Self-Contained Artefact:** All CSS, fonts, and SVG data MUST be inlined within the single HTML file. Use base64 encoding for any images if necessary. There can be no external file dependencies.

**AESTHETIC & STYLE GUIDE: "NEON DATASCAPE 2.0"**

1.  **Theme:** A sleek, high-tech dashboard from a science fiction film. The goal is maximum clarity and visual impact. Think clean lines, glowing data, and a professional, dark interface.

2.  **Layout (1400x700px):**
    * The entire design must be contained within a `1400px` by `700px` area. Use CSS Flexbox to structure content logically (e.g., a title area and a content area, or a two-column layout)!!!.
    * Use generous padding and "darkspace" to avoid clutter and guide the eye.
    * Structure information using bordered "cards" or containers to create a clear visual hierarchy.

3.  **Color Palette:**
    * **Background:** A deep midnight blue with a subtle radial gradient to add depth. Example: `radial-gradient(ellipse at center, #1A1A2E 0%, #121212 70%)`.
    * **Background Detail:** Add a faint, low-opacity grid or scanline pattern using CSS for a high-tech feel.
    * **Primary Text:** A bright, highly readable off-white (`#EAEAEA`).
    * **Accent & Glow Color:** Use a vibrant electric blue (`#00BFFF`) or magenta (`#FF00FF`) for headings, icons, borders, and key data points.
    * **Glow Effect:** Apply a subtle glow to accent elements using `text-shadow` for text and `box-shadow` for containers. Example: `text-shadow: 0 0 8px rgba(0, 191, 255, 0.8);`.

4.  **Typography:**
    * **Font:** You **must** import and use the 'Poppins' Google Font. Inline the `@import` statement in your `<style>` tag: `@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;700&display=swap');`.
    * **Headings:** `font-family: 'Poppins', sans-serif; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px;`. Use the accent color with a glow.
    * **Body Text:** `font-family: 'Poppins', sans-serif; font-weight: 300; font-size: 18px; line-height: 1.6;`.

5.  **Content & Visualization:**
    * **Be Creative:** Do not just regurgitate text. Rephrase concepts for clarity. Use icons (inlined SVG) to represent ideas.
    * **Formulas/Code:** Display mathematical formulas or key equations in a visually distinct, highlighted container to make them stand out.
    * **Diagrams:** If explaining a process or architecture, use simple divs styled to look like a flowchart or diagram.

**OUTPUT FORMAT:**

Return your answer as a plain-text list in exactly this format, with no other commentary:

===== START TAKEAWAY 1 =====
<!DOCTYPE html>
<html>
</html>
===== END TAKEAWAY 1 =====

===== START TAKEAWAY 2 =====
<!DOCTYPE html>
<html>
</html>
===== END TAKEAWAY 2 =====

(etc.)

**Paper text:**
{paper_text}
"""

PROMPT3 = """
You are a world-class expert in scientific communication and a master front-end designer specializing in high-impact, futuristic data visualization. Your task is to transform dense academic text into stunning, self-contained infographic slides.

Read the provided AI paper text and identify the 3-5 most significant takeaways. For each takeaway, craft a complete, self-contained HTML file that serves as a standalone infographic slide.

**CRITICAL DIRECTIVES:**

1.  **Engineered for Static Capture:** The final rendered HTML must be a static visual. **ABSOLUTELY NO ANIMATIONS or interactive elements.**
2.  **Self-Contained Artefact:** All CSS, fonts, and SVG data MUST be inlined within the single HTML file.

**AESTHETIC & STYLE GUIDE: "NEON DATASCAPE 2.0"**

1.  **CSS & Layout (NON-NEGOTIABLE):**
    * **The entire design MUST be strictly confined to a `1400px` by `700px` canvas.**
    * You MUST include the following CSS rules exactly as written at the top of your `<style>` tag to prevent overflow issues:
        ```css
        /* --- CRITICAL LAYOUT RULES --- */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            width: 1400px;
            height: 700px;
            overflow: hidden; /* This is a failsafe to clip any overflowing content */
            font-family: 'Poppins', sans-serif;
            background: radial-gradient(ellipse at center, #1A1A2E 0%, #121212 70%);
            color: #EAEAEA;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px;
        }}
        .container {{ /* You MUST wrap all content in a div with this class */
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        ```
    * Use CSS Flexbox or Grid for all internal layouts. Use the `gap` property for spacing.

2.  **Color Palette & Effects:**
    * **Background:** Use the gradient defined in the `body` rule above.
    * **Primary Text:** `#EAEAEA`.
    * **Accent & Glow Color:** Electric Blue (`#00BFFF`).
    * **Glow Effect:** Apply a subtle glow to accent elements. Example: `text-shadow: 0 0 8px rgba(0, 191, 255, 0.8);` and `box-shadow: 0 0 15px rgba(0, 191, 255, 0.5);`.

3.  **Typography:**
    * **Font:** You **must** import and use the 'Poppins' Google Font. Inline this statement at the top of the `<style>` tag: `@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;700&display=swap');`.
    * **Headings:** `font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #00BFFF;`.
    * **Body Text:** `font-weight: 300; font-size: 18px; line-height: 1.6;`.

4.  **Content & Visualization:**
    * Be creative and concise. Use icons (inlined SVG) and simple diagrams made from styled divs.
    * Display formulas or key terms in visually distinct, highlighted containers.

**OUTPUT FORMAT:**
Return your answer as a plain-text list in exactly this format, with no other commentary:

===== START TAKEAWAY 1 =====
<!DOCTYPE html>
<html>
</html>
===== END TAKEAWAY 1 =====

(etc.)

**Paper text:**
{paper_text}
"""

PAPER_TEXT   = """
In summary, our distinct conclusions arise from investigating a finer granularity spectrum under a different definition and ensuring appropriate training conditions for all models. Comparison with Abnar et al. ( 2025 ) . Our findings on the optimal activation ratio align with those of Abnar et al. ( 2025 ) , confirming that under a fixed compute budget, larger and sparser models yield better performance. However, our research extends beyond this conclusion in both methodology and scope.
However, our research substantially extends this direction. First, we determine training hyperparameters through extensive preliminary experiments. Second, we systematically investigate how architectural factors—particularly expert granularity and shared expert ratios—affect model performance. This reveals that beyond the primary activation ratio trend, expert granularity introduces log-polynomial adjustments to performance.
Ultimately, our primary contribution is the direct derivation of scaling laws for the efficiency leverage of MoE models relative to their dense counterparts, rather than conventional scaling laws for loss. The key advantage is its independence from specific training datasets. It directly establishes a quantitative relationship between MoE architectural configurations and their relative performance efficiency, offering more generalizable and actionable principles for model design. Comparison with Ludziejewski et al. ( 2025 ) . Our research and the work of Ludziejewski et al. ( 2025 ) are complementary, with each study addressing a distinct facet of the scaling laws for MoE models.
Our work addresses the question: given a fixed compute budget and a specific model scale ( i.e., FLOPs per token), how should one configure the architectural parameters ( i.e., expert granularity, activation ratio) to maximize performance? In contrast, their study concentrates on a different optimization problem: under the dual constraints of a compute budget and memory limitations, what is the optimal allocation of resources between model scale and data size?
While our preliminary experiments did touch upon the model-data allocation for MoE models, this exploration was intentionally limited. It was conducted under a single compute budget constraint and for a specific MoE architecture. Its primary purpose was not to derive a comprehensive allocation strategy, but rather to establish the fundamental differences in optimal resource allocation between MoE and dense models. This foundational understanding was crucial for our main experiments, as it enabled us to provision a sufficient training budget to ensure all models were compared fairly under conditions of adequate, near-optimal training. 6.2 Limitations Consistent with standard practice in scaling law research (Kaplan et al., 2020 ; Hoffmann et al., 2022 ; Clark et al., 2022 ; Ludziejewski et al., 2024 ; Abnar et al., 2025 ) , our analysis quantifies computational cost exclusively in terms of theoretical FLOPs. While FLOPs provide a valuable, hardware-agnostic metric for comparing model architectures, we acknowledge that this abstraction does not capture the full spectrum of real-world costs. Factors such as hardware specifications, system infrastructure, and implementation details can introduce discrepancies between theoretical FLOPs and actual wall-clock time.
Furthermore, due to significant resource constraints, our methodology relies on the simplifying assumption that the effects of different MoE architectural factors are largely independent. Based on this premise, we conducted a series of individual ablation studies to quantify the impact of each factor in isolation, subsequently synthesizing these effects into a unified scaling law.
We acknowledge that a primary limitation of this approach is its potential to overlook interaction effects between architectural components. Nevertheless,it remains the most pragmatic and feasible pathway within the scope of our available resources.
Despite these limitations, our findings underscore the immense potential of MoE models. By enabling a massive increase in model capacity with a minimal increase in per-token computation, they offer a clear path toward improving both model performance and efficiency. 7 Related Work 7.1 Scaling Laws for Language Models Scaling laws provide a framework for understanding and predicting the performance of language models under varying conditions. Kaplan et al. ( 2020 ) laid the foundation by demonstrating that model performance adheres to predictable power-law relationships involving model size, dataset size, and compute budget. Building on this, Hoffmann et al. ( 2022 ) introduced the Chinchilla scaling laws, highlighting the importance of balancing model size and training data volume for compute-optimal training. They showed that scaling model size without a corresponding increase in data leads to diminishing performance gains. Sardana et al. ( 2023 ) advanced this understanding by incorporating inference costs into compute-optimal frameworks, proposing strategies for optimizing performance under fixed inference constraints. Additionally, Bi et al. ( 2024 ) emphasized the critical role of data quality, demonstrating that higher-quality datasets enable more efficient scaling, particularly with larger models.
Recent advancements have applied these scaling laws to various specialized areas. For example, hyperparameter optimization has been explored in the context of scaling laws (Bi et al., 2024 ; Li et al., 2025 ) , while Gadre et al. ( 2024 ) investigated the phenomena of over-training and its implications on model performance. Furthermore, scaling laws have been analyzed for their impact on downstream task performance across a range of applications (Chen et al., 2024 ; Ruan et al., 2024 ; Isik et al., 2025 ; Hu et al., 2023 ; Grattafiori et al., 2024 ; Li et al., 2025 ) , underscoring their adaptability and relevance in addressing both theoretical and practical challenges in language modeling. 7.2 Scaling Laws for Mixture-of-Experts (MoE) Mixture-of-Experts (MoE) models (Shazeer et al., 2017 ; Lepikhin et al., 2020 ) have emerged as a powerful architecture for language modeling, primarily due to their ability to decouple computational cost from parameter count. Recent research has further explored optimizations within the MoE paradigm. For instance, DeepSeekMoE (Deepseek-AI et al., 2024 ) investigated the impact of fine-grained expert settings on model performance, proposing a novel design that incorporates shared experts and a hybrid structure combining dense layers with MoE layers. Complementing this, Zoph et al. ( 2022 ) highlighted that the performance gains from increased sparsity diminish significantly once the number of experts exceeds 256, suggesting a practical limit for highly sparse models.
With the widespread adoption of the MoE architecture, the scaling laws governing MoE models have been extensively studied. Early work by Clark et al. ( 2022 ) examined scaling by varying model size and the number of experts on a fixed dataset, concluding that routed models offer efficiency advantages only up to a certain scale. This analysis was subsequently extended by Ludziejewski et al. ( 2024 ) , who incorporated variable dataset sizes and explored the effects of expert granularity. Additionally, Wang et al. ( 2024a ) investigated the transferability and discrepancies of scaling laws between dense models and MoE models. Abnar et al. ( 2025 ) advanced this line of inquiry by deriving scaling laws for optimal sparsity, explicitly considering the interplay between training FLOPs and model size. They also analyzed the relationship between pretraining loss and downstream task performance, noting distinct behaviors between MoE and dense models on certain tasks. More recently, Ludziejewski et al. ( 2025 ) derived joint scaling laws applicable to both dense Transformers and MoE models, demonstrating that MoE architectures can outperform dense counterparts even under constraints of memory usage or total parameter count. 8 Conclusion In this work, we introduced Efficiency Leverage (EL), a metric that measures the computational advantage of an MoE model relative to a dense counterpart, to quantify the scaling behavior of MoE performance with architectural factors.
Our large-scale empirical study, based on over 300 trained models, systematically deconstructed the relationship between MoE design choices and EL. We found that the efficiency of an MoE architecture is governed by a set of predictable principles. Specifically, EL scales as a power law with both the activation ratio and the total compute budget, while expert granularity acts as a non-linear modulator with a stable optimal range. Other factors, such as shared experts, were found to have only a secondary impact.
We distilled these insights into a unified scaling law that accurately predicts the EL of any MoE configuration. The predictive power of our framework was empirically validated through the successful design and training of a 17.5B parameter MoE model, which, as predicted, achieved an efficiency leverage of over 7x compared to its dense equivalent. For future work, our framework can be extended in several key directions: (1) Incorporating memory constraints and communication overhead into the EL framework, particularly for distributed training scenarios where these factors dominate practical efficiency. (2) Developing a unified metric that balances training compute leverage with inference latency requirements, enabling end-to-end efficient architecture co-design.
We hope this work inspires continued innovation in MoE architectures, ultimately propelling the field toward greater leverage. 
"""
OUTPUT_DIR   = Path("/media/kornellewy/jan_dysk_3/auto_youtube/scraped_articles/Towards_Greater_Leverage_Scaling_Laws_for_Efficient_Mixture-of-Experts_Language_Models")
# ------------------------------------------------------------------ #

def paper_to_infographics(
    paper_text: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    have_infografic = False
    for image_path in output_dir.glob("*.jpg"):
        if "infografic_" in image_path.stem:
            have_infografic = True
            break
    if have_infografic:
        return

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY3"))
    model = genai.GenerativeModel(MODEL_NAME)

    reply = model.generate_content(PROMPT3.format(paper_text=paper_text))
    blocks = reply.text.split("===== START TAKEAWAY")[1:]

    for b in blocks:
        html_snip = b.split("===== END TAKEAWAY")[0].strip()
        num = html_snip.split("=====")[0].strip()
        html_clean = html_snip.split("</html>")[0] + "</html>"
        html_clean = html_clean.split("<!DOCTYPE html>")[1]
        out_web = output_dir / f"infografic_{num}.html"
        out_web.write_text(html_clean, encoding="utf8")
        print("saved →", out_web.resolve())
        hti = Html2Image(output_path=str(output_dir))
        hti.browser_flags = [
        "--force-device-scale-factor=2",   # 5. retina capture
        "--window-size=1500,1000",
        "--hide-scrollbars",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",]
        # out_image = output_dir / f"infografic_{num}.jpg"
        hti.screenshot(url=f"file://{out_web.resolve()}", save_as=f"infografic_{num}.jpg", size=(1500, 1000),)


if __name__ == "__main__":
    
    paper_to_infographics(PAPER_TEXT, OUTPUT_DIR)