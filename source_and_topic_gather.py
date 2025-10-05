import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re
import os
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from html2image import Html2Image
import fitz  # PyMuPDF
import requests
from pathlib import Path
from urllib.parse import urlparse
import re
from PIL import Image
import io

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
        return folder_path
    folder_path.mkdir(exist_ok=True)

    # Extract text
    text = soup.get_text(strip=True, separator=" ")

    # Remove evrythink after world rerence "References"
    refrnece_word_count = text.count('References')
    if refrnece_word_count == 1:
        text = text.split("References")[0]

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

    # Initialize html2image
    hti = Html2Image(output_path=str(folder_path))
    hti.browser_flags = ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
    i = 0
    for table in soup.find_all("table"):
        table_idx = 0
        # ⬇️ Extract the preceding <figcaption> if it's part of the table's parent <figure>
        figcaption_text = ""
        parent = table.find_parent("figure")
        if parent:
            caption = parent.find("figcaption")
            if caption:
                figcaption_text = caption.get_text(strip=True)
                if "Table" in figcaption_text:
                    can_be_table_idx = figcaption_text.split("Table ")[1].split(":")[0]
                    if can_be_table_idx.isdigit():
                        table_idx = int(can_be_table_idx)
                if table_idx == 0:
                    continue
                if table_idx > 10:
                    print(
                        f"[Warning] Skipping table {table_idx} due to excessive index."
                    )
                    continue
                # Save figcaption text to a .txt file with the same name as the image
                caption_path = folder_path / f"table_{table_idx}.txt"
                caption_path.write_text(figcaption_text, encoding="utf-8")
            else:
                continue  # Skip if no caption found

        if len(figcaption_text) < 1:
            continue

        if table_idx == 0:
            continue

        if table_idx > 10:
            print(f"[Warning] Skipping table {table_idx} due to excessive index.")
            continue

        try:
            # ✅ Validate that the table can be parsed into a DataFrame (optional safety check)
            pd.read_html(str(table))[0]

            # ✅ Prepare HTML content with basic styling and white background
            html_content = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 20px;
                    }}
                    table {{
                        border-collapse: collapse;
                        font-size: 16px;
                        width: auto;
                        max-width: 100%;
                    }}
                    th, td {{
                        border: 1px solid #333;
                        padding: 8px 12px;
                    }}
                    th {{
                        background-color: #f0f0f0;
                    }}
                    .caption {{
                        margin-top: 10px;
                        font-size: 14px;
                        font-style: italic;
                        color: #333;
                    }}
                </style>
            </head>
            <body>{str(table)}</body>
            </html>
            """

            # ✅ Save the HTML table as a PNG image
            image_filename = f"table_{table_idx}.png"
            hti.screenshot(html_str=html_content, save_as=image_filename)
            print(f"[Info] Saved table {table_idx} as image: {image_filename}")
            image_path = folder_path / image_filename

            with Image.open(image_path) as img:
                img = img.convert("RGBA")

                # Get bounding box of non-transparent content
                bbox = img.getbbox()
                if bbox:
                    # Crop the image
                    cropped = img.crop(bbox)

                    # Create white background image
                    white_bg = Image.new("RGB", cropped.size, (255, 255, 255))
                    # Paste cropped image onto white background using alpha mask
                    white_bg.paste(
                        cropped, mask=cropped.split()[3]
                    )  # use alpha channel as mask

                    # Save as non-transparent image (JPEG/PNG)
                    white_bg.save(image_path)
                    print(
                        f"[Info] Cropped image with white background saved: {image_path}"
                    )
                else:
                    print(f"[Warning] No content detected in image: {image_path}")

        except Exception as e:
            print(f"[Warning] Failed to render table {i}: {e}")

    return folder_path


def fetch_anxin_pdf_with_tables(url: str) -> list[Path]:
    tmp_pdf_path = Path(__file__).parent / "tmp.pdf"
    parsed = urlparse(url)

    # 1. Download PDF
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    # 1.5 get title
    tmp_pdf_path.write_bytes(resp.content)
    doc = fitz.open(tmp_pdf_path)
    title = ""
    for page in doc:
        title += page.get_text()
        break
    title = title.split("\n")[0]
    title = re.sub(r"[^A-Za-z0-9]", " ", title)   # keep letters/numbers
    title = re.sub(r"\s+", "_", title)            # spaces → underscores
    title = title[:100] 
    tmp_pdf_path.unlink()


    stem = re.sub(r"[^\w\-]", "_", Path(parsed.path).stem)[:60]
    output_path = OUTPUT_DIR / title
    output_path.mkdir(exist_ok=True)
    created = []

    # 2. Save PDF
    pdf_path = output_path / f"{title}.pdf"
    pdf_path.write_bytes(resp.content)
    created.append(pdf_path)

    # 3. Extract full text → article.txt
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    txt_path = output_path / "article.txt"
    txt_path.write_text(full_text, encoding="utf-8")
    created.append(txt_path)

    dpi = 150

    # 5. Embedded images
    img_counter = 0
    for page_idx in range(len(doc)):
        for img in doc.get_page_images(page_idx, full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace and pix.colorspace.n > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.alpha:
                pix = fitz.Pixmap(pix, 0)
            if pix.width < 200 or pix.height < 200:
                continue
            img_path = output_path / f"embed_img_{img_counter:03d}.png"
            pix.save(img_path)
            created.append(img_path)
            img_counter += 1

    # 6. Tables – save BOTH a crop of the table AND the entire page
    table_counter = 0
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        tabs = page.find_tables()
        for t in tabs:
            # --- crop of table only ---
            rect = t.bbox
            clip = rect
            clip = [int(c) for c in clip]
            # pix_crop = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png")))
            tab_crop_path = output_path / f"table_{table_counter:03d}_p{page_idx+1}_crop.png"
            img.crop(clip).save(tab_crop_path)
            # img.save(tab_crop_path)

            # # --- entire page (with highlight around table) ---
            # highlight = fitz.Rect(rect)
            # # page_pix = page.get_pixmap(dpi=dpi)
            # # # optional: draw a red rectangle around the table
            # page_new = doc.load_page(page_idx)  # re-load to draw
            # page_new.clip_to_rect(highlight)
            # pix_full = page_new.get_pixmap(dpi=dpi)  # re-render
            # full_path = output_path / f"table_{table_counter:03d}_p{page_idx+1}_full.png"
            # pix_full.save(full_path)
            # created.append(full_path)

            table_counter += 1


    doc.close()
    return output_path


def main_single(article_url: str):
    if "pdf" in article_url:
        folder_path = fetch_anxin_pdf_with_tables(article_url)
    else:
        folder_path = parse_article(article_url)
    print(f"Processed article: {folder_path}")
    return folder_path


if __name__ == "__main__":
    # main()
    article_url = "https://arxiv.org/pdf/2508.10975"  # Example URL
    main_single(article_url)
