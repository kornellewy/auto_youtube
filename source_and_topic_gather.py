"""
srouce sites :

https://deepseek.ai/blog
https://huggingface.co/blog
https://arxiv.org/list/cs.AI/new

Task:
python, bs4, pathlib, pandas
Cyclical (1 time per day) get the latest articles from these sites,
gether from them all text infmation tabel information and all images.
Sites are html base, arxiv have html verion of pepers
1. start by loading database csv of already procesed articles
2. load all articles from each site
3. get new articles from each site
4. get text for each article, evry image/gif/movie in evry article to image save its alt text
5. save it to output directory with article title as folder name withou emojis and special characters
6. save article ttitle without emojis and special characters to database csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re
import os
from datetime import datetime
from PIL import Image

# Constants
OUTPUT_DIR = Path("scraped_articles")
DATABASE_PATH = Path("processed_articles.csv")
SITES = {
    "deepseek": "https://deepseek.ai/blog",
    "huggingface": "https://huggingface.co/blog",
    "arxiv": "https://arxiv.org/list/cs.AI/recent",
}

# Create directories if needed
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_title(title):
    # Remove emojis and special characters
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"\s+", "_", title.strip())
    return title[:100]  # Limit filename length


def load_database():
    if DATABASE_PATH.exists():
        return pd.read_csv(DATABASE_PATH)
    return pd.DataFrame(columns=["title", "url", "date"])


def save_database(df):
    df.to_csv(DATABASE_PATH, index=False)


def fetch_soup(url):
    response = requests.get(url)
    return BeautifulSoup(response.content, "html.parser")


def extract_huggingface():
    for tag in [
        "tag=diffusion",
        "tag=nlp",
        "tag=cv",
        "tag=community",
        "tag=research",
        "tag=audio",
    ]:
        soup = fetch_soup(f"https://huggingface.co/blog?{tag}")
        for a in soup.select("a[href^='/blog/']")[:5]:
            yield f"https://huggingface.co{a['href']}"


def extract_arxiv():
    soup = fetch_soup(SITES["arxiv"])  # Pass a dummy URL for the mock
    articles = soup.select("a[title='View HTML']")
    print(f"found {len(articles)} articles on arxiv")
    for a in articles:
        yield a["href"]


def parse_article(url):
    soup = fetch_soup(url)
    title = soup.title.text if soup.title else "untitled"
    folder_name = clean_title(title)
    folder_path = OUTPUT_DIR / folder_name
    if folder_path.exists():
        return title
    folder_path.mkdir(exist_ok=True)

    # Extract text
    text = soup.get_text(strip=True, separator=" ")

    # Save plain text
    with open(folder_path / "article.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # Extract images
    for i, img in enumerate(soup.find_all("img")):
        src = img.get("src")
        if "arxiv" in url:
            src = url + "/" + src
        alt = img.get("alt", "")
        if not src or "data:image" in src:
            continue
        try:
            if src.startswith("/"):
                domain = "/".join(url.split("/")[:3])
                src = domain + src
            img_data = requests.get(src).content
            ext = Path(src).suffix or ".jpg"
            if Path(src).suffix == ".svg" or Path(src).suffix == ".jpeg":
                continue
            img_path = folder_path / f"image_{i+1}{ext}"
            with open(img_path, "wb") as f:
                f.write(img_data)
            image = Image.open(img_path)
            if image.size == (200, 200) or image.size == (0, 0):
                img_path.unlink()
                continue
            with open(folder_path / f"image_{i+1}.txt", "w", encoding="utf-8") as f:
                f.write(alt)

        except Exception as e:
            print(f"[Warning] Failed to fetch image {src}: {e}")

    return title


def main():
    db = load_database()
    new_entries = []

    for site_name, extractor in {
        # "huggingface": extract_huggingface,
        "arxiv": extract_arxiv,
    }.items():
        print(f"Processing site: {site_name}")
        for url in extractor():
            if url in db["url"].values:
                continue
            try:
                print(f"  Scraping: {url}")
                title = parse_article(url)
                new_entries.append(
                    {
                        "title": clean_title(title),
                        "url": url,
                        "date": datetime.now().isoformat(),
                    }
                )
            except Exception as e:
                print(f"[Error] Failed to process {url}: {e}")

    if new_entries:
        db = pd.concat([db, pd.DataFrame(new_entries)], ignore_index=True)
        save_database(db)
        print(f"Saved {len(new_entries)} new articles.")
    else:
        print("No new articles found.")


def main_single(article_url: str):
    title = parse_article(article_url)
    print(f"Processed article: {title}")


if __name__ == "__main__":
    # main()
    article_url = "https://arxiv.org/html/2402.09353v6"  # Example URL
    main_single(article_url)
