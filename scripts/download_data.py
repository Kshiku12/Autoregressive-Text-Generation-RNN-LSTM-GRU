import os
import urllib.request
import re

DATA_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "sherlock_holmes.txt")

def download_and_clean_text():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading dataset from: {DATA_URL}")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content = response.read().decode("utf-8-sig")

    print(f"Downloaded raw text length: {len(content):,} characters")

    # Project Gutenberg texts contain header and footer license metadata.
    # We strip the Gutenberg header and footer to preserve pure literary text.
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", content)
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", content)

    if start_match and end_match:
        text = content[start_match.end():end_match.start()].strip()
        print(f"Stripped Project Gutenberg boilerplate headers and footers.")
    else:
        text = content.strip()

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved cleaned corpus to: {RAW_PATH}")
    print(f"Corpus size: {len(text):,} characters, approx {len(text.split()):,} words.")

if __name__ == "__main__":
    download_and_clean_text()
