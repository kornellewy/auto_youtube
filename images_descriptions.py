import google.generativeai as genai
from pathlib import Path
import os
import time

from dotenv import load_dotenv

load_dotenv()


def describe_image(image_path, model_name="gemini-2.5-flash-lite"):
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(model_name)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Attempt to describe as a normal image first
        normal_prompt = "Provide a concise, high-level overview or background caption for this image."
        response_normal = model.generate_content(
            [normal_prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        normal_description = response_normal.text.strip()

        # Attempt to describe as a graph, focusing on specific details
        graph_prompt = (
            "Analyze this image as if it were a graph or chart. "
            "Identify the main topic, what is being measured or represented, how the data is presented (e.g., bar chart, line graph), "
            "and if possible, what are the approximate maximum and minimum values or trends. "
            "Conclude with a brief statement about what the graph signifies or means."
        )
        response_graph = model.generate_content(
            [graph_prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        graph_description = response_graph.text.strip()

        return normal_description, graph_description

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def process_images_in_directory(directory_path):
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]

    for image_path in directory_path.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            if image_path.stem.lower().startswith("table"):
                continue
            print(f"\n--- Processing: {image_path.name} ---")
            normal_desc, graph_desc = describe_image(image_path)
            time.sleep(30)

            if normal_desc or graph_desc:
                # Create the .txt file path with the same stem as the image
                output_txt_path = image_path.with_suffix(".txt")

                with open(output_txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Description for: {image_path.name}\n\n")
                    f.write("--- Normal Image Description (High-level overview) ---\n")
                    f.write(
                        normal_desc
                        if normal_desc
                        else "N/A - Could not generate a general description.\n"
                    )
                    f.write("\n--- Graph Description (Detailed Analysis) ---\n")
                    f.write(
                        graph_desc
                        if graph_desc
                        else "N/A - Could not generate a graph-specific description.\n"
                    )

                print(f"Descriptions saved to: {output_txt_path.name}")
            else:
                print("Could not generate any descriptions for this image.")


if __name__ == "__main__":
    image_directory = Path("/media/kornellewy/jan_dysk_3/auto_youtube/media/raw")

    process_images_in_directory(image_directory)
