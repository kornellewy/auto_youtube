import google.generativeai as genai
import cv2
from pathlib import Path
from PIL import Image


class MediaDescriptionGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def _process_image(self, file_path: Path) -> str:
        img = Image.open(file_path)
        img_data = genai.upload_file(img)
        response = self.model.generate_content(
            [
                "Describe this image in detail, focusing on key objects, colors, and the overall context. Be concise but descriptive.",
                img_data,
            ]
        )
        genai.delete_file(img_data.name)
        return response.text.strip()

    def _process_gif(self, file_path: Path) -> str:
        gif = Image.open(file_path)

        frames = []
        try:
            while True:
                frames.append(gif.copy())
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        if not frames:
            return "Unable to extract frames from GIF."

        # Select a few key frames to represent the GIF
        selected_frames = [frames[0]]
        if len(frames) > 1:
            if len(frames) > 2:
                selected_frames.append(frames[len(frames) // 2])  # Middle frame
            selected_frames.append(frames[-1])  # Last frame

        input_parts = [
            "Describe this GIF, focusing on the actions, objects, and overall narrative portrayed across these frames. Be concise and descriptive."
        ]
        for frame in selected_frames:
            frame_data = genai.upload_file(frame)
            input_parts.append(frame_data)

        response = self.model.generate_content(input_parts)

        # Clean up uploaded files
        for frame_data in input_parts[1:]:  # Skip the initial text prompt
            genai.delete_file(frame_data.name)

        return response.text.strip()

    def _process_mp4(self, file_path: Path) -> str:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return "Could not open video file."

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return "Video file contains no frames."

        # Sample frames (e.g., 5 frames evenly spaced)
        num_samples = min(frame_count, 5)
        frames_to_read = (
            [int(i * (frame_count - 1) / (num_samples - 1)) for i in range(num_samples)]
            if num_samples > 1
            else [0]
        )

        extracted_frames = []

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frames_to_read:
                # Convert OpenCV BGR to RGB PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                extracted_frames.append(pil_img)

        cap.release()

        if not extracted_frames:
            return "No frames extracted from video."

        input_parts = [
            "Describe this video, focusing on the key scenes, objects, actions, and overall narrative or content. Be concise and descriptive."
        ]
        uploaded_files = []
        for frame_img in extracted_frames:
            frame_data = genai.upload_file(frame_img)
            input_parts.append(frame_data)
            uploaded_files.append(frame_data)

        response = self.model.generate_content(input_parts)

        for uploaded_file in uploaded_files:
            genai.delete_file(uploaded_file.name)

        return response.text.strip()

    def generate_descriptions(self, directory: Path):
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # Add more as needed
        gif_extensions = {".gif"}

        for item_path in directory.rglob("*"):
            if item_path.is_file():
                stem = item_path.stem
                description_filepath = item_path.with_suffix(".txt")

                if description_filepath.exists():
                    continue

                description = None
                if item_path.suffix.lower() in image_extensions:
                    description = self._process_image(item_path)
                elif item_path.suffix.lower() in gif_extensions:
                    description = self._process_gif(item_path)
                elif item_path.suffix.lower() in video_extensions:
                    description = self._process_mp4(item_path)

                if description:
                    try:
                        description_filepath.write_text(description, encoding="utf-8")
                        print(f"Generated description for {item_path.name}")
                    except Exception as e:
                        print(f"Error writing description for {item_path.name}: {e}")


if __name__ == "__main__":
    # Set your Google API Key as an environment variable or replace 'YOUR_GOOGLE_API_KEY'
    # Example: os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
    import os

    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # Create dummy media files for testing
    test_media_dir = Path("test_media")
    test_media_dir.mkdir(parents=True, exist_ok=True)

    # Dummy image
    try:
        Image.new("RGB", (60, 30), color="red").save(test_media_dir / "dummy_image.png")
    except Exception as e:
        print(f"Could not create dummy image: {e}. Ensure Pillow is installed.")

    # Dummy GIF (requires Pillow to create)
    try:
        frames = []
        for i in range(5):
            img = Image.new("RGB", (60, 30), color=(i * 50, 0, 255 - i * 50))
            frames.append(img)
        frames[0].save(
            test_media_dir / "dummy_gif.gif",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
    except Exception as e:
        print(f"Could not create dummy GIF: {e}. Ensure Pillow is installed.")

    # Dummy MP4 (requires OpenCV to create, more complex)
    # This part is illustrative; creating a real MP4 without external tools is non-trivial.
    # For testing, you'd typically place an existing MP4 file in 'test_media_dir'.
    # For example, you could download a short sample video.
    # If you want to create a blank MP4, it's still complex.
    # For demonstration, we'll assume a dummy MP4 is placed there manually or downloaded.
    # Example placeholder:
    # (test_media_dir / 'dummy_video.mp4').write_text("DUMMY_VIDEO_CONTENT", encoding='utf-8')
    # This won't work as a real video, but will create the file.
    print(
        "\nReminder: For MP4 testing, place a real .mp4 file in 'test_media' directory."
    )

    generator = MediaDescriptionGenerator(api_key=API_KEY)
    generator.generate_descriptions(directory=test_media_dir)
