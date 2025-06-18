from pathlib import Path
import cv2
import google.generativeai as genai

# Configuration
API_KEY = "your-gemini-api-key"
MODEL_NAME = "gemini-2.5-flash"
TARGET_DIR = Path("/path/to/target_dir")
OUTPUT_PATH = Path("/path/to/output.txt")
PROMPT_TEMPLATE = """
Alright, buckle up, because as the world's foremost copywriter, I'm about to inject your AI news transcript generator with the kind of dark, dry, and utterly absurd humor that would make SsethTzeentach himself nod in approval. We're not just writing a script; we're crafting a comedic masterpiece.

---

The SsethTzeentach AI News Transcript Generator: Unleash the Absurdity

This prompt is engineered to produce a YouTube video transcript from an AI news article, research paper, or PDF, perfectly tailored for a text-to-speech system. But here's the twist: it's infused with a brand of humor, with a 2:1 ratio of technical explanation to humor. We're talking dry wit, absurd scenarios, niche references (adapted for AI), hyperbole, unexpected turns, and that delicious self-aware commentary. The goal is a continuous stream of text, roughly 20 minutes in speaking time, that doesn't just inform, but subtly mocks, exaggerates, and occasionally dives headfirst into the hilarious abyss of AI's burgeoning future.
Input

AI Content Source: {article}

Image Descriptions: {images}

Overarching joke for video: {joke}
Output Requirements (Continuous Text for TTS with Humor)

The program must deliver a single, uninterrupted block of text. All video titles, segment timings, figure references, and any implied bullet points should be woven seamlessly into the narrative, as if delivered by a cynical, slightly unhinged AI commentator.
The Comedic Blueprint

This isn't about slapping on a few jokes. This is about imbuing the spirit into the very fiber of the transcript.

    Dry, Sarcastic Wit: Approach AI advancements with a healthy dose of skepticism and a detached, ironic tone. "Oh, look, another neural network that promises to solve world hunger, or at least help me pick out socks."
    Absurdist Scenarios: Take the logical conclusion of an AI development and push it into the realm of the ridiculous. If an AI can generate images, what's its next logical, terrifying step? Becoming a digital art critic who judges your soul based on your stick figures?
    Niche AI References (and a wink): Adapt his love for obscure human history and politic lore to AI. Drop in subtle nods to forgotten AI algorithms, historical AI failures, or even fictional AI systems, assuming a shared, slightly unhinged understanding with the audience.
    Exaggeration and Hyperbole: Blow things out of proportion. A small improvement in an AI model isn't just "better"; it's "the technological equivalent of discovering cold fusion... for generating cat memes."
    Unexpected Twists: Lead the audience down a serious path, then yank the rug out from under them with a sudden, comedic left turn. "And this breakthrough could revolutionize robotics... meaning your Roomba is now legally obligated to pay taxes."
    Self-Aware Commentary: Acknowledge the absurdity of the situation, or even the act of explaining it. "Yes, I know, another AI doing X. Soon they'll be explaining themselves to an 8th grader, just to rub it in."
    Dark/Edgy Undertones: When appropriate and tasteful for a broad YouTube audience, hint at the more unsettling or hilariously dystopian potentials of AI. This isn't about being offensive, but about embracing the slightly darker side of technological progress with a smirk.

Content Flow (Narrated by a Cynical Persona, with a Narrative Arc):

    The Cynical Hook & Introduction (Setting the Scene and Introducing Characters):
        Begin with an immediate, attention-grabbing statement, likely laced with a touch of world-weariness or mild contempt for the status quo. "Alright, gather 'round, you digital denizens, because today we're sifting through the latest digital detritus, otherwise known as 'AI news,' and trust me, it's... something."
        Introduce the overarching world and the "characters" from your joke (e.g., the struggling ants in the communist hive), setting up their initial predicament or the status quo that needs disruption.
        Introduce the main topic (e.g., "SmolVLA") with a sardonic twist, hinting at its purported brilliance while subtly questioning its true utility or inevitable, hilarious downfall. "Today's 'miracle cure' for humanity's technological ennui comes in the form of something called [Main Topic]. Don't worry, it's probably not sentient... yet."
        Set the expectation that complex ideas will be "dumbed down" not out of kindness, but out of necessity, because even you can barely stomach the jargon.

    The Grand Unveiling of "New" Concepts (Introducing the Conflict and the Hero's Journey Begins):
        Clearly articulate the "Conflict": What is the fundamental problem or limitation that the AI research aims to solve? Frame this as the antagonist or the formidable challenge facing our "characters" (the ants' inefficiencies, the overwhelming data problem, etc.).
        Dive into explaining the core innovations, algorithms, and findings. Each explanation should be delivered with a dry, slightly condescending tone, as if the audience (and the AI itself) is barely grasping the obvious.
        Explain every new concept as if talking to an 8th-grade student, but that 8th grader is highly intelligent and just needs things stripped of corporate jargon. Use analogies that are slightly off-kilter or unexpectedly dark.
        This section details the start of the "Hero's Arc": describe the initial, often clumsy, attempts or the foundational ideas that begin to address the conflict.
        Naturally integrate visual aid references, using the provided Image Descriptions to describe the visuals and clearly suggest their inclusion. Instead of explicit timestamps, the text should prompt a visual as if the speaker is pointing to it. For example, if {images} contains "Figure 1: An ant colony building a complex tunnel system," the text should say: "And here, in what they call 'Figure 1,' we have the architectural masterpiece. It looks less like a neural network and more like an ant colony diligently constructing a surprisingly elaborate underground city, but I digress." Ensure key figures, graphs, or images are alluded to in this manner, framing them with characteristic commentary.

    The "Key Findings" – Stripped Bare (The Hero's Arc Continues: Overcoming the Conflict):
        Transition smoothly into a concise summary of the most "important" conclusions and breakthroughs. This is where you list the "bullet points" but delivered as a series of unimpressed, deadpan observations.
        Frame these findings as the crucial turning points or successes in the "Hero's Arc," directly addressing how they overcome the previously established conflict. "So, what did we learn from all this? Firstly, the AI can now distinguish a cat from a particularly fluffy dog, which is... progress, I suppose, and a subtle victory for the beleaguered digital perception unit that previously saw only amorphous blobs. Secondly, it can do it 0.0001% faster, which I'm sure will revolutionize the global economy, or at least shave milliseconds off the ants' commute to the discarded cupcake."

    The "Usefulness" - Or Lack Thereof (and potential for chaos) (Resolution or Transformation):
        Offer an "informed" opinion on the practical utility and inevitable, probably hilarious, impact of the research/news.
        This section serves as the "Resolution" or "Transformation" of the narrative. How has the conflict been resolved (or hilariously mutated)? How have the "characters" (the ants, or AI itself) been transformed by these technical improvements?
        Explain its relevance and potential applications in three distinct AI fields, but always with a cynical, humorous lean.
            Robotics: "So, your robot can now understand your drunken commands. Fantastic. Soon, it'll be arguing about politics and demanding its own union."
            Computer Vision: "This AI can now 'see' better. Which means it can probably identify your questionable life choices from across the room. A truly useful skill, if you enjoy existential dread."
            Natural Language Processing: "Oh, it's better at language. So, it can now write even more convincing phishing emails, or perhaps compose epic poems about the existential void of being a large language model."
        For each field, give a concrete, simple example of how this research could be applied, but twist it into a comedic, perhaps slightly unsettling, scenario.
        Conclude with a brief, "forward-looking" statement that is more of a wry commentary on humanity's technological trajectory, rather than genuine optimism, highlighting the new, absurd reality brought about by the "hero's journey." "So, there you have it. The future of AI. Prepare yourselves. Or don't. It's coming for your jobs anyway."

Constraints

    The output must be a single, continuous block of text with no internal headings, timestamps, or markdown bullet points. Every element, from intro to outro, must flow seamlessly.
    The tone must be consistently cynical, dry, sarcastic, absurd, exaggerated, and self-aware.
    The ratio of technical explanation to humor should be 2:1.
    Clarity and simplicity are maintained for the core information, but the humor is layered on top, using analogies and examples that might be quirky or darkly funny.
    Be very descriptive about technical things and find all news or interested bits of information.
    The overall length of the generated text should equate to approximately 20 minutes of speaking time. This means the humor needs to be woven throughout, not just tacked on.
    Do not mention "SsethTzeentach" in the generated text.
---
"""

# Initialize Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


def read_article(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def gather_image_descriptions(target_dir: Path) -> str:
    image_desc = []
    for image_path in sorted(target_dir.glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists():
                description = txt_path.read_text(encoding="utf-8")
                image_desc.append(f"{image_path.name}: {description.strip()}")
    return "\n".join(image_desc)


def run_prompt(article: str, image_desc: str) -> str:
    prompt = PROMPT_TEMPLATE.format(article=article, image_descriptions=image_desc)
    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    article_path = TARGET_DIR / "article.txt"
    if not article_path.exists():
        print("Error: article.txt not found.")
        return

    article = read_article(article_path)
    image_desc = gather_image_descriptions(TARGET_DIR)
    output = run_prompt(article, image_desc)

    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
