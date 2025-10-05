import nltk
from pathlib import Path
import re


class VideoScriptGenerator:
    def __init__(
        self,
        default_photo: str = "default_photo",
        default_voice: str = "narrator",
        default_anim: str = "zoomin",
        default_transitionin: str = "none",
        default_transitionout: str = "none",
        last_anim: str = "fadeout",
        last_transitionout: str = "fadeout",
    ):

        self.default_photo = default_photo
        self.default_voice = default_voice
        self.default_anim = default_anim
        self.default_transitionin = default_transitionin
        self.default_transitionout = default_transitionout
        self.last_anim = last_anim
        self.last_transitionout = last_transitionout

    @staticmethod
    def load_text_content(filepath: Path) -> str | None:
        try:
            content = filepath.read_text(encoding="utf-8")
            return content
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None

    def _split_text_to_sentences(self, text_content: str) -> list[str]:
        cleaned_sentences = []
        for sentence in text_content.split("\n"):
            sentence = sentence.strip()
            sentence = re.sub(r"\s+", " ", sentence)
            if len(sentence) > 5:
                cleaned_sentences.append(sentence)
        return cleaned_sentences

    def _suggest_keywords(self, sentence: str) -> list[str]:
        keywords = []
        return keywords

    def generate(self, input_filepath: Path, output_filepath: Path):
        try:
            text_content = input_filepath.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_filepath}")
            return
        except Exception as e:
            print(f"Error reading input file: {e}")
            return

        sentences = self._split_text_to_sentences(text_content)
        script_lines = []
        num_sentences = len(sentences)

        for i, sentence in enumerate(sentences):
            current_anim = self.default_anim
            current_transitionin = self.default_transitionin
            current_transitionout = self.default_transitionout
            current_photo = self.default_photo

            # Apply specific rules for first/last sentences
            if i == 0:
                current_anim = "zoomin"
                current_transitionin = "none"
                current_transitionout = "none"
            elif i == num_sentences - 1:
                current_anim = self.last_anim
                current_transitionout = self.last_transitionout
                current_transitionin = self.default_transitionin

            # Content-based photo selection logic

            current_animation = "stationary"
            if sentence.startswith("***"):
                current_photo = "###heading"
                current_transitionin = "fadein"
                current_transitionout = "fadeout"
                current_animation = "stationary"

            control_line = (
                f"[Photo: {current_photo}] "
                f"[Voice: {self.default_voice}] "
                f"[Anim: {current_animation}] "
                f"[transitionin: {current_transitionin}] "
                f"[transitionout: {current_transitionout}]"
            )
            script_lines.append(control_line)
            script_lines.append(sentence)

            script_lines.append("")

        try:
            output_filepath.write_text("\n".join(script_lines), encoding="utf-8")
            print(f"Video script saved to {output_filepath}")
        except Exception as e:
            print(f"Error writing output file: {e}")


if __name__ == "__main__":
    # Ensure NLTK Punkt tokenizer is downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        nltk.download("punkt")

    generator = VideoScriptGenerator(
        default_photo="alan_turing1", default_anim="zoomin"
    )

    # Define file paths
    input_dir = Path("./projects/raw_scripts")
    output_dir = Path("./projects/scripts")

    input_file = (
        input_dir
        / "24_07_2025_DeepSeek_LLM_Scaling_Open-Source_Language_Models_with_Longtermism.txt"
    )
    output_file = (
        output_dir
        / "24_07_2025_DeepSeek_LLM_Scaling_Open-Source_Language_Models_with_Longtermism.txt"
    )

    input_text_content = generator.load_text_content(input_file)

    # Write dummy content to input file
    input_file.write_text(input_text_content, encoding="utf-8")
    print(f"Dummy input file created at: {input_file}")

    # Initialize and run the generator

    generator.generate(input_filepath=input_file, output_filepath=output_file)

    print(f"\nCheck the generated script at: {output_file}")
