import json
import os
from pathlib import Path

import google.generativeai as genai

LLM_PROMPT_TEMPLATE = """
You are an expert content summarizer and keyword extractor. Your task is to process the following article text.
Break the text into logical, meaningful chunks, primarily at the sentence or 3 sentence max, ensuring that no natural technical thought piece is split. For each chunk, identify a concise list of 5-10 relevant keywords that accurately describe its content.

Return the output as a JSON array of objects, where each object has two keys:
- 'text_chunk': The extracted text chunk.
- 'keywords': A JSON array of strings representing the keywords for that chunk.

Example output format:
```json
[
{
    "text_chunk": "First sentence or chunk.",
    "keywords": ["keyword1", "keyword2"]
},
{
    "text_chunk": "Second sentence or chunk, potentially longer if it's a cohesive thought.",
    "keywords": ["keyword3", "keyword4", "keyword5"]
}
]
```

Article Text to Process:
{article_text}

Strictly adhere to the JSON output format. Do not include any additional commentary or text outside the JSON array.
"""
MODEL = "gemini-2.0-lite"


class VideoScriptGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(MODEL)

    def _process_text_with_llm(self, text_content: str) -> list[dict]:
        prompt = LLM_PROMPT_TEMPLATE.format(article_text=text_content)
        try:
            response = self.model.generate_content(prompt)
            # The LLM's response.text might be a JSON string, need to parse it.
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            print(f"LLM response was not valid JSON: {e}")
            print(f"Raw LLM response: {response.text}")
            return []
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return []

    def generate(self, input_filepath: Path, output_filepath: Path):
        try:
            text_content = input_filepath.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Input file not found at {input_filepath}")
            return
        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        llm_processed_chunks = self._process_text_with_llm(text_content)
        script_lines = []

        for chunk_data in llm_processed_chunks:
            text_chunk = chunk_data.get("text_chunk", "").strip()
            keywords = ", ".join(chunk_data.get("keywords", []))

            if text_chunk:  # Ensure there's actual text
                control_line = (
                    f"[Media: none] [Voice: none] [Anim: none] "
                    f"[transitionin: none] [transitionout: none] [PaperImg: none] "
                    f"[Keywords: {keywords}]"
                )
                script_lines.append(control_line)
                script_lines.append(text_chunk)
                script_lines.append("")  # Blank line for separation

        try:
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            output_filepath.write_text("\n".join(script_lines), encoding="utf-8")
            print(f"Video script saved to {output_filepath}")
        except Exception as e:
            print(f"Error writing output file: {e}")


if __name__ == "__main__":
    # Set your Google API Key as an environment variable or replace 'YOUR_GOOGLE_API_KEY'
    # Example: os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set or provided.")

    # Define file paths
    input_dir = Path("./projects/raw_scripts")
    output_dir = Path("./projects/scripts")

    input_file_name = "08_06_2025_haggingface_smolvam.txt"
    input_file_path = input_dir / input_file_name
    output_file_path = output_dir / input_file_name  # Keep same name in output dir

    # Create dummy content for the input file for demonstration
    dummy_article_content = """
Alright, gather 'round, you digital denizens, because today we're sifting through the latest digital detritus, otherwise known as 'AI news,' and trust me, it's... something. Today's 'miracle cure' for humanity's technological ennui comes in the form of something called **SmolVLA**. Don't worry, it's probably not sentient... yet. We're about to delve into the nitty-gritty of this miniature marvel, stripping away the corporate jargon and presenting it in a way that even your intellectually gifted but perpetually distracted eighth-grade cousin could grasp. Because, let's be honest, even I struggle with some of these acronyms, and I've seen things, digital things, that would make a circuit board weep.

Now, let's get into the guts of this supposed revolution. **SmolVLA**, or "Small Vision-Language-Action" model, is Hugging Face's attempt to democratize robotics. The grand problem with most **Vision-Language-Action models**, or **VLAs**, is that they're absolute behemoths. We're talking billions of parameters, requiring server racks that probably hum louder than a dying refrigerator and consume enough electricity to power a small nation. This makes them prohibitively expensive to train and deploy, relegating advanced robotics to institutions with budgets larger than some small countries' GDP. Basically, if you don't have a spare few million sitting around for computing power, your robot dreams remain just that: dreams. SmolVLA, on the other hand, aims to shrink that footprint without sacrificing performance. It's like taking a supercomputer and stuffing it into a particularly sturdy lunchbox, then realizing that lunchbox now has a crippling gambling addiction, much like a worker ant in a communist hive trying to sneak an extra crumb.
    """

    # Ensure input directory exists and write dummy content
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file_path.write_text(dummy_article_content.strip(), encoding="utf-8")
    print(f"Dummy input file created at: {input_file_path.resolve()}")

    # Initialize and run the generator
    generator = VideoScriptGenerator(api_key=api_key)

    print(
        f"Generating script from '{input_file_path.name}' to '{output_file_path.name}'..."
    )
    try:
        generator.generate(
            input_filepath=input_file_path, output_filepath=output_file_path
        )
        print(f"Script generation complete.")
    except Exception as e:
        print(f"An error occurred during script generation: {e}")
