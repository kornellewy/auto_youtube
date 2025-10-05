import random
from pathlib import Path
import time
import yaml
import shutil
import json

from source_and_topic_gather import main_single as source_and_topic_gather_function
from scripts_writer import main as scripts_writer_function
from script_preprocesor import VideoScriptGenerator
from video_creator2 import main as main_function
from video_creator2 import create_title_slide
from config_utils import load_or_create_config, save_config
from yt_title_and_video_description import YTMetaFactory
from yt_video_upload import YoutubeUploader
from yt_shudler import generate_calendar
from yt_thumbnail_scorer import ThumbnailScorer
from yt_script_scorer import WriterScorer


### pipline do dodaniwa nowych memwow
# https://pypi.org/project/ddgs/#2-images
# https://pypi.org/project/ddgs/ i video
# preprocess_images after webscraping
# helpers/image_post_pocesing_after_webscraping.py

# create image descriptions
# images_descriptions.py

### pipline robienie filmow
# create configs for every video

# iterate over all config in configs

# source_and_topic_gather.py
all_papers = [
    # dodanie prezentacji na starcie ktora szybko miw i cachy co jest w filmie
    "https://arxiv.org/html/2509.14353v1",
    # "https://arxiv.org/html/2509.26507v1",
    # https://arxiv.org/html/2509.25454v2
    # https://arxiv.org/html/2509.22622v1
    
]
papers_dir_paths = []
for paper in all_papers:
    folder_path = source_and_topic_gather_function(paper)
    papers_dir_paths.append(Path(folder_path))
# scripts_writer.py
jokes_strings = [
    """
    DreamControl: diffusion models and Reinforcement Learning conect together like power rengers creating super robot.
    """,
    # """
    # Braine inspired ai architerute, that all archteutes are ? 
    # """
    # """
    # using monte carlo menthod like in chesst, so leangue is some kind of game ? 
    # """,
    # """
    # ai steven speaberg
    # """

]


write_scripts_paths = []
for folder_path in papers_dir_paths:
    output_script_path = Path(folder_path) / f"{Path(folder_path).stem}.txt"
    scripts_writer_function(
        target_dir=folder_path,
        output_path=output_script_path,
        joke=random.choice(jokes_strings),
    )
    write_scripts_paths.append(output_script_path)
    time.sleep(30)  # Sleep to avoid rate limiting
# # script_preprocesor.py
generator = VideoScriptGenerator(default_photo="alan_turing1", default_anim="zoomin")
converted_scripts_paths = []
for script_path in write_scripts_paths:
    output_file = script_path.parent / f"{Path(script_path).stem}_procesed.txt"
    if output_file.exists():
        print(f"Output file {output_file} already exists, skipping.")
        converted_scripts_paths.append(output_file)
        continue
    input_text_content = generator.load_text_content(script_path)
    generator.generate(input_filepath=script_path, output_filepath=output_file)
    converted_scripts_paths.append(output_file)

# score script yt_script_scorer.py
writer_scorer = WriterScorer()
for script_path in write_scripts_paths:
    writer_scorer_result_path = Path(script_path).parent / "writer_scorer_result.json"
    if writer_scorer_result_path.exists():
        print(
            f"Writer scorer result {writer_scorer_result_path} already exists, skipping."
        )
        continue
    script_text = Path(script_path).read_text()
    paper_text = (Path(script_path).parent / "article.txt").read_text()
    result = writer_scorer.score_script(script_text, paper_text)
    result = json.loads(result)
    new_result = {}
    for k, v in result.items():
        if v is None or v == "None":
            v = 0
        new_result[k] = v
    result = new_result

    with open(writer_scorer_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

# create title, description, tags
factory = YTMetaFactory()
for script_path in write_scripts_paths:
    factory.save_metadata(script_path, Path(script_path).parent)

# TODO: thumbnails creation to papers directories
# filler
for folder_path in papers_dir_paths:
    thumbnail_path = folder_path / "thumbnail.jpg"
    if thumbnail_path.exists():
        continue
    title = (folder_path / "title.txt").read_text(encoding="utf-8")
    thumbnail = create_title_slide(
        text=title,
        size=(1280, 720),
        save_path=thumbnail_path,
    )


outro_videos_paths = []
for idx, script_path in enumerate(converted_scripts_paths):
    video_output_path = script_path.parent / "final_video.mp4"
    if video_output_path.exists():
        continue
    main_function(
        script_path=script_path,
        downloaded_article_dir_path=papers_dir_paths[idx],
        outro_video_path=None,
        output_video_path=video_output_path,
    )
    # main.py
    # TODO: ADD pdf from anvix
    # TODO: ADD difrent sources
    # TODO: ADD option for multiple sources mixing
    # TODO: ADD deepreaserch thing for script writing
    # TODO: ADD MORE MEMES mei.py, beter meme choing
# raise
# upload
yt_uploader = YoutubeUploader()
calendar = generate_calendar(["monday", "friday"], look_ahead_days=30)
latest_dates = [
    str(date.isoformat()).replace("T00:00:00+00:00", "T12:00:00.0Z")
    for date in calendar
]
for script_path_idx, script_path in enumerate(write_scripts_paths):
    video_path = list(script_path.parent.glob("*mp4"))[0]
    thumbnail_path = script_path.parent / "thumbnail.jpg"
    title = (script_path.parent / "title.txt").read_text(encoding="utf-8")
    description = (script_path.parent / "description.txt").read_text(encoding="utf-8")
    tags = (script_path.parent / "hashtags.txt").read_text(encoding="utf-8")
    tags = ",".join(tags.split())
    publish_at = latest_dates[script_path_idx]

    yt_uploader.upload(
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        title=title,
        description=description,
        tags=tags,
        publish_at=publish_at,
    )
