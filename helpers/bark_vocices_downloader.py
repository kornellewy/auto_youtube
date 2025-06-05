import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse, unquote, urljoin
import time
import re
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# The URL of the Notion page to scrape
NOTION_PAGE_URL = "https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c"
# Base directory where voice files will be saved
OUTPUT_BASE_DIR = "suno_ai_voices"


def sanitize_directory_name(name: str) -> str:
    """Sanitizes a string to be used as a directory name."""
    # Remove or replace characters not suitable for directory names
    name = re.sub(r"[^\w\s-]", "", name)  # Keep alphanumeric, whitespace, hyphen
    name = re.sub(r"[-\s]+", "_", name).strip(
        "_"
    )  # Replace whitespace/hyphens with underscore
    return name if name else "misc"


def download_file(url: str, dest_folder: Path, filename_hint: str) -> bool:
    """
    Downloads a file from a URL to a destination folder.
    Derives filename from URL path if hint is not a valid .npz filename.
    """
    actual_filename = ""
    if filename_hint and filename_hint.lower().endswith(".npz"):
        actual_filename = Path(filename_hint).name  # Ensure it's just the name
    else:
        try:
            parsed_url = urlparse(url)
            path_filename = Path(unquote(parsed_url.path)).name
            if path_filename and path_filename.lower().endswith(".npz"):
                actual_filename = path_filename
            else:
                logging.warning(
                    f"Could not determine a clear .npz filename from URL path: {url} (Hint: '{filename_hint}'). Skipping."
                )
                return False
        except Exception as e:
            logging.error(f"Error parsing filename from URL {url}: {e}. Skipping.")
            return False

    if not actual_filename:
        logging.warning(f"Filename for {url} is empty. Skipping.")
        return False

    dest_file_path = dest_folder / actual_filename
    dest_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading {actual_filename} from {url} to {dest_file_path}...")
    try:
        response = requests.get(
            url, stream=True, timeout=60
        )  # Increased timeout for larger files
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        with open(dest_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded {actual_filename}")
        return True
    except requests.exceptions.Timeout:
        logging.error(f"Timeout while downloading {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {url}: {e}")
    except Exception as e:
        logging.error(f"An error occurred while saving {actual_filename}: {e}")

    # If download failed, attempt to remove partial file
    if dest_file_path.exists():
        try:
            dest_file_path.unlink()
            logging.info(f"Removed partially downloaded file: {dest_file_path}")
        except OSError as e:
            logging.error(
                f"Error removing partially downloaded file {dest_file_path}: {e}"
            )
    return False


def scrape_suno_voices(page_url: str, output_base_dir_str: str = "suno_ai_voices"):
    """
    Scrapes .npz voice files from the Suno AI Notion page and organizes them by language.
    """
    logging.info(f"Starting scraper for Suno AI voices from: {page_url}")
    base_output_path = Path(output_base_dir_str)
    base_output_path.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(page_url, headers=headers, timeout=30)
        response.raise_for_status()
        logging.info(f"Successfully fetched page: {page_url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Fatal error fetching page {page_url}: {e}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    # --- Parsing Logic ---
    # This is the most critical and potentially fragile part.
    # We try to find all links to .npz files.
    # Then, for each link, we navigate up its parent elements to find a "card" or "item" container.
    # Within that container, we search for the "Language" property and its value.

    npz_links = soup.find_all("a", href=re.compile(r"\.npz(\?|$)", re.IGNORECASE))
    logging.info(f"Found {len(npz_links)} potential .npz links. Analyzing each...")

    if not npz_links:
        logging.warning(
            "No .npz links found. The page structure might have changed, or content is loaded dynamically "
            "in a way that `requests` cannot capture (e.g. needs JavaScript execution). "
            "Consider inspecting the page with browser developer tools."
        )
        return

    downloaded_count = 0
    for link_tag in npz_links:
        npz_url_str = link_tag.get("href")
        if not npz_url_str:
            continue

        # Ensure URL is absolute
        npz_url = urljoin(page_url, npz_url_str)

        # Filename hint from link text
        filename_hint = link_tag.get_text(strip=True)

        language = "Unknown"  # Default language

        # Try to find the language associated with this link.
        # Heuristic: Traverse up from the link to find a common ancestor ("card") for the voice entry.
        # Then search for "Language" property within that card.
        # Notion's structure is complex; this search depth is arbitrary.
        current_element = link_tag
        card_ancestor = None
        for _ in range(6):  # Try up to 6 levels (adjust if needed)
            if current_element.parent:
                current_element = current_element.parent
                card_ancestor = (
                    current_element  # Consider this current parent as a potential card
                )
                # Check if this card_ancestor contains a "Language" label
                if card_ancestor.find(string=re.compile(r"^\s*Language\s*$", re.I)):
                    break  # Found a promising ancestor
            else:  # Reached top (e.g., `soup` object itself)
                card_ancestor = soup
                break

        if not card_ancestor:  # Should not happen if soup is fallback
            card_ancestor = soup

        # Now, within this card_ancestor, look for the language.
        # Find elements that contain the text "Language" (case-insensitive).
        lang_label_elements = card_ancestor.find_all(
            string=re.compile(r"^\s*Language\s*$", re.I)
        )

        for label_el in lang_label_elements:
            # Assumption: The actual language value is often in the next significant element
            # relative to the element containing the "Language" label.
            # This could be a sibling of `label_el.parent` or a sibling of `label_el` itself,
            # or a specific child if the structure is nested.

            # Try looking at the parent of the label and its next sibling.
            # E.g., <div>Language</div> <div class="value">ENGLISH</div>
            # Here, label_el is "Language", label_el.parent is the first div.
            # Its next_sibling is the second div containing "ENGLISH".
            if label_el.parent:
                value_container = label_el.parent.find_next_sibling()
                if value_container:
                    lang_text = value_container.get_text(separator=" ", strip=True)
                    # Basic validation: not too long, not another label
                    if (
                        lang_text
                        and 0 < len(lang_text) < 50
                        and not re.match(
                            r"^(Gender|Sample|Bark|Voice)", lang_text, re.I
                        )
                    ):
                        language = lang_text
                        break  # Found language

            # Add more strategies here if the above fails, e.g., checking label_el.find_next_sibling() directly, etc.
            # This part is highly dependent on the exact HTML structure.

        if language == "Unknown":
            logging.warning(
                f"Could not reliably determine language for {npz_url} (filename hint: '{filename_hint}'). Saving to 'Unknown' folder."
            )

        language_cleaned_for_dir = sanitize_directory_name(language)
        lang_dir_path = base_output_path / language_cleaned_for_dir

        if download_file(npz_url, lang_dir_path, filename_hint):
            downloaded_count += 1

        time.sleep(0.25)  # Be polite to the server, small delay between downloads

    logging.info(
        f"\nScraping finished. Attempted to download files for {len(npz_links)} links."
    )
    logging.info(f"Successfully downloaded {downloaded_count} files.")

    if downloaded_count == 0 and len(npz_links) > 0:
        logging.warning(
            "No files were successfully downloaded. Check previous error messages and the HTML structure of the page."
        )
    elif len(npz_links) == 0:
        logging.info("No .npz links were found on the page to begin with.")

    logging.info(
        f"Files are saved in subdirectories under: {base_output_path.resolve()}"
    )
    logging.info("\n--- Important Reminder ---")
    logging.info(
        "This scraper's success depends on the Notion page's current HTML structure."
    )
    logging.info(
        "If Notion changes its page structure, this script (especially the parsing logic for finding links and languages) may need to be updated."
    )
    logging.info(
        "You might need to inspect the page with browser developer tools and adjust BeautifulSoup selectors accordingly."
    )


if __name__ == "__main__":
    scrape_suno_voices(NOTION_PAGE_URL, OUTPUT_BASE_DIR)
