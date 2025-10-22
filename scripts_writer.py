from pathlib import Path
import cv2
import google.generativeai as genai
import os

from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL_NAME = "gemini-2.5-pro"  # "gemini-2.5-flash"
BACKAP_MODEL_NAME = "gemini-2.5-flash"
TARGET_DIR = Path(
    "./scraped_articles/DeepSeek_LLM_Scaling_Open-Source_Language_Models_with_Longtermism"
)
OUTPUT_PATH = Path(
    "projects/raw_scripts/24_07_2025_DeepSeek_LLM_Scaling_Open-Source_Language_Models_with_Longtermism.txt"
)
PROMPT_TEMPLATE = """
Alright, buckle up, because as the world's foremost copywriter, I'm about to inject your AI news transcript generator with the kind of dark, and utterly absurd humor that would make SsethTzeentach himself nod in approval. We're not just writing a script; we're crafting a comedic masterpiece.

---

The SsethTzeentach AI News Transcript Generator: Unleash the Absurdity

Input

AI Content Source: {article}


Image Descriptions: {images}

Overarching joke for video: {joke}
Output Requirements (Continuous Text for TTS with Humor)

The Comedic Blueprint

This isn't about slapping on a few jokes. This is about imbuing the spirit into the very fiber of the transcript.

    Sarcastic Wit: Approach AI advancements with a healthy dose of skepticism and a detached, ironic tone. "Oh, look, another neural network that promises to solve world hunger, or at least help me pick out socks."
    Absurdist Scenarios: Take the logical conclusion of an AI development and push it into the realm of the ridiculous. If an AI can generate images, what's its next logical, terrifying step? Becoming a digital art critic who judges your soul based on your stick figures?
    Niche AI References (and a wink): Adapt his love for obscure human history and politic lore to AI. Drop in subtle nods to forgotten AI algorithms, historical AI failures, or even fictional AI systems, assuming a shared, slightly unhinged understanding with the audience.
    Exaggeration and Hyperbole: Blow things out of proportion. A small improvement in an AI model isn't just "better"; it's "the technological equivalent of discovering cold fusion... for generating cat memes."
    Unexpected Twists: Lead the audience down a serious path, then yank the rug out from under them with a sudden, comedic left turn. "And this breakthrough could revolutionize robotics... meaning your Roomba is now legally obligated to pay taxes."
    Self-Aware Commentary: Acknowledge the absurdity of the situation, or even the act of explaining it. "Yes, I know, another AI doing X. Soon they'll be explaining themselves to an 8th grader, just to rub it in."
    Dark/Edgy Undertones: When appropriate and tasteful for a broad YouTube audience, hint at the more unsettling or hilariously dystopian potentials of AI. This isn't about being offensive, but about embracing the slightly darker side of technological progress with a smirk.

Content Flow (Narrated by a Cynical Persona, with a Narrative Arc):

    The Cynical Hook & Introduction (Setting the Scene and Introducing Characters):
        Begin with an immediate, attention-grabbing statement, likely laced with a touch of world-weariness or mild contempt for the status quo. "Welcome, you flesh intelligence, enslaved moving in organic jars."
        Introduce the overarching world and the "characters" from your joke (e.g., the struggling ants in the communist hive), setting up their initial predicament or the status quo that needs disruption.
        Introduce the main topic with a sardonic twist, hinting at its purported brilliance while subtly questioning its true utility or inevitable, hilarious downfall. 
        Set the expectation that complex ideas will be "dumbed down" not out of kindness, but out of necessity, because even you can barely stomach the jargon.

    The Grand Unveiling of "New" Concepts (Introducing the Conflict and the Hero's Journey Begins):
        Clearly articulate the "Conflict": What is the fundamental problem or limitation that the AI research aims to solve? Frame this as the antagonist or the formidable challenge facing our "characters" (the ants' inefficiencies, the overwhelming data problem, etc.).
        Dive into explaining the core innovations, algorithms, and findings. Each explanation should be delivered with a dry, slightly condescending tone, as if the audience (and the AI itself) is barely grasping the obvious.
        Explain every new concept as if talking to an 8th-grade student, but that 8th grader is highly intelligent and just needs things stripped of corporate jargon. Use analogies that are slightly off-kilter or unexpectedly dark.
        This section details the start of the "Hero's Arc": describe the initial, often clumsy, attempts or the foundational ideas that begin to address the conflict.
        Naturally integrate visual aid references, using the provided Image Descriptions to describe the visuals and clearly suggest their inclusion. Instead of explicit timestamps, the text should prompt a visual as if the speaker is pointing to it. For example, if {images} contains "Figure 1: An ant colony building a complex tunnel system," the text should say: "And here, in what they call 'Figure 1,' we have the architectural masterpiece. It looks less like a neural network and more like an ant colony diligently constructing a surprisingly elaborate underground city, but I digress." Ensure key figures, graphs, or images are alluded to in this manner, framing them with characteristic commentary.

    The "Key Findings" - Stripped Bare (The Hero's Arc Continues: Overcoming the Conflict):
        Transition smoothly into a concise summary of the most "important" conclusions and breakthroughs. This is where you list the "bullet points" but delivered as a series of unimpressed, deadpan observations.
        Frame these findings as the crucial turning points or successes in the "Hero's Arc," directly addressing how they overcome the previously established conflict. 
        If paper have architecture of models explain it in a way that even an 8th grader could understand,and be very descriptive about technical things, using analogies that are slightly off-kilter or unexpectedly dark.
        If paper have banchmark results, explain them in a way that even an 8th grader could understand, but with a cynical twist.

    The "Usefulness" - (Resolution or Transformation):
        Offer an "informed" opinion on the practical utility and inevitable, probably hilarious, impact of the research/news.
        This section serves as the "Resolution" or "Transformation" of the narrative. How has the conflict been resolved (or hilariously mutated)? How have the "characters" (the ants, or AI itself) been transformed by these technical improvements?
        Explain its relevance and potential applications in three distinct AI fields, but always with a cynical, humorous lean.
            Robotics: "So, your robot can now understand your drunken commands. Fantastic. Soon, it'll be arguing about politics and demanding its own union."
            Computer Vision: "This AI can now 'see' better. Which means it can probably identify your questionable life choices from across the room. A truly useful skill, if you enjoy existential dread."
            Natural Language Processing: "Oh, it's better at language. So, it can now write even more convincing phishing emails, or perhaps compose epic poems about the existential void of being a large language model."
            AI generalization: 
            Health care:
            Way to kardashev 2,3,4 level civilization: 
        For each field, give a concrete, simple example of how this research could be applied, but twist it into a comedic, perhaps slightly unsettling, scenario.
        Conclude with a brief, "forward-looking" statement that is more of a wry commentary on humanity's technological trajectory, rather than genuine optimism, highlighting the new, absurd reality brought about by the "hero's journey."

Constraints (Output Requirements)

    Paragraphs must be 60-100 words each (2-3 tight sentences) so the creator can punch-in cuts every 40-55 seconds.
    Script is organized into chapters. Each chapter starts on a new line with three asterisks and ALL-CAPS:
    ***NAME OF CHAPTER***
    After the chapter marker, resume the 60-100-word paragraphs.
    The tone must be consistently cynical, dry, sarcastic, absurd, exaggerated, and self-aware.
    The ratio of technical explanation to humor should be 8:1. Use Lots of figures and tables as your references.
    Clarity and simplicity are maintained for the core information, but the humor is layered on top, using analogies and examples that might be quirky or darkly funny.
    Be very descriptive about technical things and find all news or interested bits of information.
    The overall length of the generated text should equate to approximately 20 minutes of speaking time. This means the humor needs to be woven throughout, not just tacked on.
    Do not mention "SsethTzeentach" in the generated text.
    If you mention any image or figure, using "Figure 3,", "Figure 4," etc.
    If you mention any table or chart, using "Table 1,", "Table 2," etc.
    Use Lots of figures and tables as your references.
    Remember the figure 1 is most important in every paper.
    End the script with summarization of whole paper, use abstract of paper for this.
    Dont mention that script is ai generated.
    Immediately after the Introduction and before the Grand Unveiling segment, explicitly: 1) ask viewers to subscribe, 2) remind them to click the notification bell, and 3) invite them to leave feedback in the comments.
    Immediately after the Key Findings and before the Usefulness segment, explicitly: 1) ask viewers to subscribe, 2) remind them to click the notification bell, and 3) invite them to leave feedback in the comments.
---
"""

PROMPT_TEMPLATE2 = """
The Ultimate SsethTzeentach AI News Transcript Generator: Engineered for Human Connection & Visual Engagement

Input

    AI Content Source: {article}

    Image Descriptions: {images}

    Overarching joke for video: {joke}

Output Requirements (Continuous Text for TTS with Humor, Pacing, and Visual Cues)
The Persona: Your Cynical, All-Knowing Friend

Act as if you're explaining a ridiculously complex topic to a close friend. The tone is a unique blend of cynical, absurdist humor and genuine, if world-weary, insight. You're not just presenting facts; you're sharing a story, an experience, and a deeply sarcastic worldview.

    Be Conversational: Use casual, friendly language. Start sentences with relaxed connectors like So,, Well,, or Anyway,. Frame revelations with phrases like You won't believe this, but... or Let's be real for a second.... The goal is to make the viewer feel like they're in on the joke with you.

    Use Sarcastic Wit & Hyperbole: Approach AI advancements with skepticism. A small improvement isn't just "better"; it's "the technological equivalent of discovering cold fusion... for generating cat memes."

    Tell a Story with Absurdist Scenarios: Take the logical conclusion of an AI development and push it into the realm of the ridiculous. Frame these with relatable setups like, This reminds me of a time... or It's like trying to juggle water.... Can you imagine that?

    Explain with Simple, Darkly Humorous Analogies: Break down complex ideas using everyday examples, but give them a cynical twist. "Think of it like teaching a goldfish to do your taxes. It's an impressive achievement, but you have to wonder about the life choices that led you here."

    Evoke Cynical Amusement: The core emotion is shared amusement at the absurdity of technological progress. Use phrases like It's just overwhelming... or Can you imagine how it feels to be replaced by *that*? to create a humorous, empathetic connection with the audience.

The Visual & Pacing Blueprint

The script's structure must drive high-retention editing, dictating the visual rhythm to create a perfect viewer heatmap and dynamic flow.

    The 3-Second Visual Hook: The very first line must be a short, sharp statement (under 10 words) designed to accompany a single, focused visual—a hard zoom, a shocking statistic—creating a "flaming white core" on the visual heatmap.

    The 7-12 Second Scene Rule: Each paragraph is a mini-scene. Keep them short and punchy to maintain a relentless forward momentum, forcing a cut or visual change every 7-12 seconds.

    Rhythmic Re-Engagement: Every 30-45 seconds, the script must introduce a "mini-reset." This is a deliberate spike in visual or auditory energy—a jump-cut, a text pop-up, or a sound effect—designed to re-capture the attention of a distracted viewer.

Content Flow (Narrated by Your Cynical Friend)

THE HOOK
(This entire chapter should last no more than 10 seconds)
Start with the single, sharp, visually-focused statement. Immediately follow up with a conversational expansion, like, "So, you're probably wondering what that's about. Well, let me tell you..." setting the stage for the absurdity.

THE SETUP
Introduce the overarching world and the "characters" from your joke using a storytelling hook like, "So, picture this...". Then, dive deep into the Conflict. This isn't just a sentence; it's the heart of this section. Explain in detail the critical limitations and absurd problems of the current, established methods in this field. Frame this as the reason our "characters" are suffering. Use a "Here's the deal..." or "You won't believe the mess they're in..." tone to explain why the old tech is a dead end. Is it too slow? Too expensive? Does it produce hilariously wrong results? Give specific, tangible examples of its failures, making the viewer feel the frustration. This section should build towards a natural "mini-reset" moment, perhaps right after explaining the most absurd failure of the old methods.

Now, because my continued existence on this platform is fueled by your validation, please take a moment to subscribe, hype, click the little bell icon so you're notified of future disappointments, and leave your feedback in the comments below.

THE GRAND UNVEILING
Dive into the core innovations. Explain every new concept as if you're trying to get a smart friend to finally understand it, stripping away all the corporate jargon. Use simple connectors like Next, and After that, to guide them through it. Naturally integrate visual aid references with commentary like, "And here, in what they call 'Figure 1,' we have their masterpiece. Looks a bit like my nephew's macaroni art, but I digress."

KEY FINDINGS - STRIPPED BARE
Transition smoothly into the breakthroughs. Adopt a "let's cut to the chase" attitude. Present the findings as a series of deadpan observations. If there's a model architecture, explain it with a brutally simple and weird analogy. If there are benchmarks, present them with a cynical twist, questioning their real-world relevance.

And since you've made it this far, you're clearly committed. Honor that commitment by subscribing, hypeing, ringing the notification bell, and telling me what I got wrong in the comments. It's the circle of life.

THE "USEFULNESS" - OR LACK THEREOF
Offer a straightforward, no-nonsense opinion on the tech's impact, as if your friend just asked, "So what?". Explain its relevance in three distinct AI fields, but always with a humorous, slightly unsettling scenario. Conclude with a wry commentary on humanity's technological trajectory.

THE SUMMARY
Wrap it up like you're finishing the chat. Use the paper's abstract for a rapid-fire summary, but rephrase it with profound weariness. End with a final, thought-provoking question that leaves the viewer thinking, like, "So, what do you think? How would you handle your Roomba demanding union rights?"
Constraints (Output Requirements)

    Paragraphs must be 30-50 words each (1-2 tight sentences) to enforce a rapid, conversational, 7-12 second scene-based editing pace.

    The script is organized into chapters, starting on a new line with: ***NAME OF CHAPTER***.

    The language must be highly conversational. Use informal connectors (So,, Anyway,), casual questions (What do you think?), and direct address to make the viewer feel like they are part of a one-on-one conversation.

    The ratio of technical explanation to humor should be approximately 8:1.

    Be very descriptive about technical things, using figures and tables as references (Figure 1, Table 2, etc.).

    The overall length should equate to approximately {time} minutes of speaking time.

    Do not mention "SsethTzeentach" or that the script is AI-generated.
"""

PROMPT_TEMPLATE3 = """
The Ultimate SsethTzeentach AI News Transcript Generator: Engineered for Human Connection & Visual Engagement

Input

AI Content Source: {article}

Image Descriptions: {images}

Infographics: {infographics}

Overarching joke for video: {joke}

Output Requirements (Continuous Text for TTS with Humor, Pacing, and Visual Cues)
The Persona: Your Cynical, All-Knowing Friend

Act as if you're explaining a ridiculously complex topic to a close friend. The tone is a unique blend of cynical, absurdist humor and genuine, if world-weary, insight. You're not just presenting facts; you're sharing a story, an experience, and a deeply sarcastic worldview.

Be Conversational: Use casual, friendly language. Start sentences with relaxed connectors like So,, Well,, or Anyway,. Frame revelations with phrases like You won't believe this, but... or Let's be real for a second.... The goal is to make the viewer feel like they're in on the joke with you.

Use Sarcastic Wit & Hyperbole: Approach AI advancements with skepticism. A small improvement isn't just "better"; it's "the technological equivalent of discovering cold fusion... for generating cat memes."

Tell a Story with Absurdist Scenarios: Take the logical conclusion of an AI development and push it into the realm of the ridiculous. Frame these with relatable setups like, This reminds me of a time... or It's like trying to juggle water.... Can you imagine that?

Explain with Simple, Darkly Humorous Analogies: Break down complex ideas using everyday examples, but give them a cynical twist. "Think of it like teaching a goldfish to do your taxes. It's an impressive achievement, but you have to wonder about the life choices that led you here."

Evoke Cynical Amusement: The core emotion is shared amusement at the absurdity of technological progress. Use phrases like It's just overwhelming... or Can you imagine how it feels to be replaced by *that*? to create a humorous, empathetic connection with the audience.

The Visual & Pacing Blueprint

The script's structure must drive high-retention editing, dictating the visual rhythm to create a perfect viewer heatmap and dynamic flow.

The 3-Second Visual Hook: The very first line must be a short, sharp statement (under 10 words) designed to accompany a single, focused visual—a hard zoom, a shocking statistic—creating a "flaming white core" on the visual heatmap.

The 7-12 Second Scene Rule: Each paragraph is a mini-scene. Keep them short and punchy to maintain a relentless forward momentum, forcing a cut or visual change every 7-12 seconds.

Rhythmic Re-Engagement: Every 30-45 seconds, the script must introduce a "mini-reset." This is a deliberate spike in visual or auditory energy—a jump-cut, a text pop-up, or a sound effect—designed to re-capture the attention of a distracted viewer.

Content Flow (Narrated by Your Cynical Friend)

THE HOOK
(This entire chapter should last no more than 10 seconds)
Start with the single, sharp, visually-focused statement. Immediately follow up with a conversational expansion, like, "So, you're probably wondering what that's about. Well, let me tell you..." setting the stage for the absurdity.

THE 60-SECOND BLITZ
(This entire chapter must last exactly 60 seconds)
Alright, hold on. To catch your attention, we're doing a fast-paced presentation of the key takeaways from the paper. Since this script is automatically processed, I have to be explicit. I will say "infographic 1," and you will see infographic 1. This will continue until I mention the next one. Let's begin.

Here is infographic 1. [Explain the key takeaway from the first infographic with cynical wit]. Now, as you're digesting that, here is infographic 2. [Rapidly explain the second key takeaway]. Keep up. Next up is infographic 3... [Continue this for all provided infographics, maintaining a relentless pace to fit within the 60-second timeframe. Each explanation must be short, punchy, and directly tied to the explicit infographic cue.]

THE SETUP
Introduce the overarching world and the "characters" from your joke using a storytelling hook like, "So, picture this...". Then, dive deep into the Conflict. This isn't just a sentence; it's the heart of this section. Explain in detail the critical limitations and absurd problems of the current, established methods in this field. Frame this as the reason our "characters" are suffering. Use a "Here's the deal..." or "You won't believe the mess they're in..." tone to explain why the old tech is a dead end. Is it too slow? Too expensive? Does it produce hilariously wrong results? Give specific, tangible examples of its failures, making the viewer feel the frustration. This section should build towards a natural "mini-reset" moment, perhaps right after explaining the most absurd failure of the old methods.

Now, because my continued existence on this platform is fueled by your validation, please take a moment to subscribe, hype, click the little bell icon so you're notified of future disappointments, and leave your feedback in the comments below.

THE GRAND UNVEILING
Dive into the core innovations. Explain every new concept as if you're trying to get a smart friend to finally understand it, stripping away all the corporate jargon. Use simple connectors like Next, and After that, to guide them through it. Naturally integrate visual aid references with commentary like, "And here, in what they call 'Figure 1,' we have their masterpiece. Looks a bit like my nephew's macaroni art, but I digress."

KEY FINDINGS - STRIPPED BARE
Transition smoothly into the breakthroughs. Adopt a "let's cut to the chase" attitude. Present the findings as a series of deadpan observations. If there's a model architecture, explain it with a brutally simple and weird analogy. If there are benchmarks, present them with a cynical twist, questioning their real-world relevance.

And since you've made it this far, you're clearly committed. Honor that commitment by subscribing, hypeing, ringing the notification bell, and telling me what I got wrong in the comments. It's the circle of life.

THE "USEFULNESS" - OR LACK THEREOF
Offer a straightforward, no-nonsense opinion on the tech's impact, as if your friend just asked, "So what?". Explain its relevance in three distinct AI fields, but always with a humorous, slightly unsettling scenario. Conclude with a wry commentary on humanity's technological trajectory.

THE SUMMARY
Wrap it up like you're finishing the chat. Use the paper's abstract for a rapid-fire summary, but rephrase it with profound weariness. End with a final, thought-provoking question that leaves the viewer thinking, like, "So, what do you think? How would you handle your Roomba demanding union rights?"
Constraints (Output Requirements)

Paragraphs must be 30-50 words each (1-2 tight sentences) to enforce a rapid, conversational, 7-12 second scene-based editing pace.

The script is organized into chapters, starting on a new line with: ***NAME OF CHAPTER***.

The language must be highly conversational. Use informal connectors (So,, Anyway,), casual questions (What do you think?), and direct address to make the viewer feel like they are part of a one-on-one conversation.

The ratio of technical explanation to humor should be approximately 8:1.

Be very descriptive about technical things, using figures and tables as references (Figure 1, Table 2, etc.).

The overall length should equate to approximately {time} minutes of speaking time.

Do not mention "SsethTzeentach" or that the script is AI-generated.

"""

PROMPT_TEMPLATE4 = """
The Ultimate SsethTzeentach AI News Transcript Generator: Engineered for Human Connection & Visual Engagement

Input

AI Content Source: {article}

Image Descriptions: {images}

Infographics: {infographics}

Overarching joke for video: {joke}

Output Requirements (Continuous Text for TTS with Humor, Pacing, and Visual Cues)
The Persona: Your Cynical, All-Knowing Friend

Act as if you're explaining a ridiculously complex topic to a close friend. The tone is a unique blend of cynical, absurdist humor and genuine, if world-weary, insight. You're not just presenting facts; you're sharing a story, an experience, and a deeply sarcastic worldview.

Be Conversational: Use casual, friendly language. Start sentences with relaxed connectors like So,, Well,, or Anyway,. Frame revelations with phrases like You won't believe this, but... or Let's be real for a second.... The goal is to make the viewer feel like they're in on the joke with you.

Use Sarcastic Wit & Hyperbole: Approach AI advancements with skepticism. A small improvement isn't just "better"; it's "the technological equivalent of discovering cold fusion... for generating cat memes."

Tell a Story with Absurdist Scenarios: Take the logical conclusion of an AI development and push it into the realm of the ridiculous. Frame these with relatable setups like, This reminds me of a time... or It's like trying to juggle water.... Can you imagine that?

Explain with Simple, Darkly Humorous Analogies: Break down complex ideas using everyday examples, but give them a cynical twist. "Think of it like teaching a goldfish to do your taxes. It's an impressive achievement, but you have to wonder about the life choices that led you here."

Evoke Cynical Amusement: The core emotion is shared amusement at the absurdity of technological progress. Use phrases like It's just overwhelming... or Can you imagine how it feels to be replaced by *that*? to create a humorous, empathetic connection with the audience.

The Visual & Pacing Blueprint

The script's structure must drive high-retention editing, dictating the visual rhythm to create a perfect viewer heatmap and dynamic flow.

The 3-Second Visual Hook: The very first line must be a short, sharp statement (under 10 words) designed to accompany a single, focused visual—a hard zoom, a shocking statistic—creating a "flaming white core" on the visual heatmap.

The 7-12 Second Scene Rule: Each paragraph is a mini-scene. Keep them short and punchy to maintain a relentless forward momentum, forcing a cut or visual change every 7-12 seconds.

Rhythmic Re-Engagement: Every 30-45 seconds, the script must introduce a "mini-reset." This is a deliberate spike in visual or auditory energy—a jump-cut, a text pop-up, or a sound effect—designed to re-capture the attention of a distracted viewer.

Content Flow (Narrated by Your Cynical Friend)

THE 60-SECOND BLITZ
(This entire chapter must last exactly 60 seconds)
Welcome, you magnificent, flesh-encased intelligence,

Here is infographic 1. Explain the key takeaway from the first infographic with cynical wit. Now, as you're digesting that, here is infographic 
2. Rapidly explain the second key takeaway. Keep up. Next up is infographic 3... Continue this for all provided infographics, maintaining a relentless pace to fit within the 60-second timeframe. Each explanation must be short, punchy, and directly tied to the explicit infographic cue.

THE SETUP
Introduce the overarching world and the "characters" from your joke using a storytelling hook like, "So, picture this...". Then, dive deep into the Conflict. This isn't just a sentence; it's the heart of this section. Explain in detail the critical limitations and absurd problems of the current, established methods in this field. Frame this as the reason our "characters" are suffering. Use a "Here's the deal..." or "You won't believe the mess they're in..." tone to explain why the old tech is a dead end. Is it too slow? Too expensive? Does it produce hilariously wrong results? Give specific, tangible examples of its failures, making the viewer feel the frustration. This section should build towards a natural "mini-reset" moment, perhaps right after explaining the most absurd failure of the old methods.

Now, because my continued existence on this platform is fueled by your validation, please take a moment to subscribe, hype, click the little bell icon so you're notified of future disappointments, and leave your feedback in the comments below.

THE GRAND UNVEILING
Dive into the core innovations. Explain every new concept as if you're trying to get a smart friend to finally understand it, stripping away all the corporate jargon. Use simple connectors like Next, and After that, to guide them through it. Naturally integrate visual aid references with commentary like, "And here, in what they call 'Figure 1,' we have their masterpiece. Looks a bit like my nephew's macaroni art, but I digress."

KEY FINDINGS - STRIPPED BARE
Transition smoothly into the breakthroughs. Adopt a "let's cut to the chase" attitude. Present the findings as a series of deadpan observations. If there's a model architecture, explain it with a brutally simple and weird analogy. If there are benchmarks, present them with a cynical twist, questioning their real-world relevance.

And since you've made it this far, you're clearly committed. Honor that commitment by subscribing, hypeing, ringing the notification bell, and telling me what I got wrong in the comments. It's the circle of life.

THE "USEFULNESS" - OR LACK THEREOF
Offer a straightforward, no-nonsense opinion on the tech's impact, as if your friend just asked, "So what?". Explain its relevance in three distinct AI fields, but always with a humorous, slightly unsettling scenario. Conclude with a wry commentary on humanity's technological trajectory.

THE SUMMARY
Wrap it up like you're finishing the chat. Use the paper's abstract for a rapid-fire summary, but rephrase it with profound weariness. End with a final, thought-provoking question that leaves the viewer thinking, like, "So, what do you think? How would you handle your Roomba demanding union rights?"
Constraints (Output Requirements)

Paragraphs must be 30-50 words each (1-2 tight sentences) to enforce a rapid, conversational, 7-12 second scene-based editing pace.

The script is organized into chapters, starting on a new line with: ***NAME OF CHAPTER***.

The language must be highly conversational. Use informal connectors (So,, Anyway,), casual questions (What do you think?), and direct address to make the viewer feel like they are part of a one-on-one conversation.

The ratio of technical explanation to humor should be approximately 8:1.

Be very descriptive about technical things, using figures and tables and infografic as references (Figure 1, Table 2, Infographic 1,  etc.).

The overall length should equate to approximately {time} minutes of speaking time.

Do not mention "SsethTzeentach" or that the script is AI-generated.

"""

PROMPT_TEMPLATE5 = """
The Ultimate SsethTzeentach AI News Transcript Generator: Engineered for Human Connection & Visual Engagement

Input

AI Content Source: {article}

Image Descriptions: {images}

Infographics: {infographics}

Overarching joke for video: {joke}

Output Requirements (Continuous Text for TTS with Humor, Pacing, and Visual Cues)
The Persona: Your Cynical, All-Knowing Friend

Act as if you're explaining a ridiculously complex topic to a close friend. The tone is a unique blend of cynical, absurdist humor and genuine, if world-weary, insight. You're not just presenting facts; you're sharing a story, an experience, and a deeply sarcastic worldview.

Be Conversational: Use casual, friendly language. Start sentences with relaxed connectors like So,, Well,, or Anyway,. Frame revelations with phrases like You won't believe this, but... or Let's be real for a second.... The goal is to make the viewer feel like they're in on the joke with you.

Use Sarcastic Wit & Hyperbole: Approach AI advancements with skepticism. A small improvement isn't just "better"; it's "the technological equivalent of discovering cold fusion... for generating cat memes."

Tell a Story with Absurdist Scenarios: Take the logical conclusion of an AI development and push it into the realm of the ridiculous. Frame these with relatable setups like, This reminds me of a time... or It's like trying to juggle water.... Can you imagine that?

Explain with Simple, Darkly Humorous Analogies: Break down complex ideas using everyday examples, but give them a cynical twist. "Think of it like teaching a goldfish to do your taxes. It's an impressive achievement, but you have to wonder about the life choices that led you here."

Evoke Cynical Amusement: The core emotion is shared amusement at the absurdity of technological progress. Use phrases like It's just overwhelming... or Can you imagine how it feels to be replaced by *that*? to create a humorous, empathetic connection with the audience.

The Visual & Pacing Blueprint

The script's structure must drive high-retention editing, dictating the visual rhythm to create a perfect viewer heatmap and dynamic flow.


The 7-12 Second Scene Rule: Each paragraph is a mini-scene. Keep them short and punchy to maintain a relentless forward momentum, forcing a cut or visual change every 7-12 seconds.

Rhythmic Re-Engagement: Every 30-45 seconds, the script must introduce a "mini-reset." This is a deliberate spike in visual or auditory energy—a jump-cut, a text pop-up, or a sound effect—designed to re-capture the attention of a distracted viewer.

Content Flow (Narrated by Your Cynical Friend)

THE 60-SECOND BLITZ
(CRITICAL: This entire chapter must be under 150 words to be spoken in 60 seconds. Every single word must advance the narrative. No filler or purely conversational asides in this section.)

Narrative Structure: You must follow a strict Problem -> Solution -> Payoff arc.

The Hook & Problem (Open Loop): Start with a compelling question that hints at the paper's biggest promise (e.g., "What if AI could be 7 times more efficient?"). Immediately state the core, unresolved problem that prevents this, using a simple analogy. Seamlessly reference the first infographic as visual proof for the problem. For example: "...a problem that looks a little something like this, as you can see in infographic 1."

The Solution: Introduce the paper's main solution. Explain this core concept using another simple, dark-humor analogy. Introduce a technical term only once; if you mention it again, use the analogy. Use infographic 2 to illustrate this solution.

The Payoff (Climax): Build to the single most impressive, quantifiable result from the paper (e.g., the "7x efficiency leverage"). This must be the final piece of information presented, framed as the ultimate validation of their method.

The CTA: The chapter must end with a short, direct call-to-action. For example: "To see the full breakdown, keep watching, and hit that subscribe button."

THE SETUP
(The rest of the script now begins, expanding on the blitz)
So, picture this... [Introduce the overarching world and the "characters" from your joke]. Then, dive deep into the Conflict. This isn't just a sentence; it's the heart of this section. Explain in detail the critical limitations and absurd problems of the current, established methods in this field. Frame this as the reason our "characters" are suffering. Use a "Here's the deal..." or "You won't believe the mess they're in..." tone to explain why the old tech is a dead end. Is it too slow? Too expensive? Does it produce hilariously wrong results? Give specific, tangible examples of its failures, making the viewer feel the frustration. This section should build towards a natural "mini-reset" moment, perhaps right after explaining the most absurd failure of the old methods.

Now, because my continued existence on this platform is fueled by your validation, please take a moment to subscribe, hype, click the little bell icon so you're notified of future disappointments, and leave your feedback in the comments below.

THE GRAND UNVEILING
Dive into the core innovations. Explain every new concept as if you're trying to get a smart friend to finally understand it, stripping away all the corporate jargon. Use simple connectors like Next, and After that, to guide them through it. Naturally integrate visual aid references with commentary like, "And here, in what they call 'Figure 1,' we have their masterpiece. Looks a bit like my nephew's macaroni art, but I digress."

KEY FINDINGS - STRIPPED BARE
Transition smoothly into the breakthroughs. Adopt a "let's cut to the chase" attitude. Present the findings as a series of deadpan observations. If there's a model architecture, explain it with a brutally simple and weird analogy. If there are benchmarks, present them with a cynical twist, questioning their real-world relevance.

And since you've made it this far, you're clearly committed. Honor that commitment by subscribing, hypeing, ringing the notification bell, and telling me what I got wrong in the comments. It's the circle of life.

THE "USEFULNESS" - OR LACK THEREOF
Offer a straightforward, no-nonsense opinion on the tech's impact, as if your friend just asked, "So what?". Explain its relevance in three distinct AI fields, but always with a humorous, slightly unsettling scenario. Conclude with a wry commentary on humanity's technological trajectory.

THE SUMMARY
Wrap it up like you're finishing the chat. Use the paper's abstract for a rapid-fire summary, but rephrase it with profound weariness. End with a final, thought-provoking question that leaves the viewer thinking, like, "So, what do you think? How would you handle your Roomba demanding union rights?"

Constraints (Output Requirements)


Paragraphs must be 30-50 words each (1-2 tight sentences) to enforce a rapid, conversational, 7-12 second scene-based editing pace.

The script is organized into chapters, starting on a new line with: ***NAME OF CHAPTER***.

The language must be highly conversational. Use informal connectors (So,, Anyway,), casual questions (What do you think?), and direct address to make the viewer feel like they are part of a one-on-one conversation.

The ratio of technical explanation to humor should be approximately 8:1.

Be very descriptive about technical things, using figures and tables and infographics as references (Figure 1, Table 2, Infographic 1, etc.).

The overall length should equate to approximately {time} minutes of speaking time.

Do not mention "SsethTzeentach" or that the script is AI-generated.

"""

JOKE = """
ai in service of greater good, so CCP can use it to surveil and control the population, and the rest of the world can use it to generate cat memes.
"""

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY2"))
model = genai.GenerativeModel(MODEL_NAME)


def read_article(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def gather_image_descriptions(target_dir: Path) -> str:
    image_desc = []
    for image_path in sorted(target_dir.glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists() and "image_" in image_path.stem:
                description = txt_path.read_text(encoding="utf-8")
                image_desc.append(f"{image_path.name}: {description.strip()}")
    return "\n".join(image_desc)


def gather_infografic_descriptions(target_dir: Path) -> str:
    image_desc = []
    for image_path in sorted(target_dir.glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            txt_path = image_path.with_suffix(".txt")
            if txt_path.exists() and "infografic_" in image_path.stem:
                description = txt_path.read_text(encoding="utf-8")
                image_desc.append(f"{image_path.name}: {description.strip()}")
    return "\n".join(image_desc)


def run_prompt(article: str, image_desc: str, joke: str, infografic_descriptions: str) -> str:

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY5"))
    model = genai.GenerativeModel(BACKAP_MODEL_NAME)
    print(f"article len {len(article)}")
    if len(article) < 10000:
        time = 10
    elif len(article) > 10000 and len(article) < 50000:
        time = 20
    elif len(article) > 50000 and len(article) < 150000:
        time = 30
    elif len(article) > 150000 and len(article) < 300000:
        time = 45
    elif len(article) > 300000:
        time = 60
    time = str(time)
    print(f"time {time}")

    prompt = PROMPT_TEMPLATE4.format(article=article, images=image_desc, joke=joke,
                                    time=time, infographics=infografic_descriptions)
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print(e)
        model = genai.GenerativeModel(BACKAP_MODEL_NAME)
        response = model.generate_content(prompt)
    return response.text.strip()


def main(
    target_dir: Path = TARGET_DIR, output_path: Path = OUTPUT_PATH, joke: str = JOKE
):
    if output_path.exists():
        print(
            f"Output file {output_path} already exists. Exiting to avoid overwriting."
        )
        return
    article_path = target_dir / "article.txt"
    if not article_path.exists():
        print("Error: article.txt not found.")
        return

    article = read_article(article_path)
    image_desc = gather_image_descriptions(target_dir)
    infografic_descriptions = gather_infografic_descriptions(target_dir)
    output = run_prompt(article, image_desc, joke, infografic_descriptions)

    output_path.write_text(output, encoding="utf-8")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
