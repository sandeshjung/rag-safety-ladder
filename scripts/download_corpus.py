"""Download the hospital trust's HR policies that the employer corpus came from.

Fetches every PDF linked under "Human Resources policies and procedures" on
the trust's public policies page. This is how corpus/employer/ was built, but
it will not reproduce it exactly: the corpus is a hand-picked selection of 23
of these documents, and the page changes as policies are revised. The PDFs
committed in corpus/employer/ are the ones every result was measured on, and
corpus/register.json pins their exact contents.

Usage:
  uv run python scripts/download_corpus.py                 # into corpus/downloads/
  uv run python scripts/download_corpus.py --out some/dir
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://www.kgh.nhs.uk/policies-and-procedures/"
SECTION = "human resources policies and procedures"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def clean_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename).strip()


def hr_pdf_links() -> list[str]:
    """PDF links between the HR section heading and the next heading."""
    response = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    heading = next(
        (
            h
            for h in soup.find_all(["h2", "h3"])
            if SECTION in h.get_text(" ", strip=True).lower()
        ),
        None,
    )
    if heading is None:
        raise RuntimeError("HR section not found; the page structure may have changed.")

    links: list[str] = []
    for element in heading.find_all_next():
        if element.name in ("h2", "h3"):
            break
        href = element.get("href") if element.name == "a" else None
        if not href:
            continue
        url = urljoin(BASE_URL, str(href))
        if urlparse(url).path.lower().endswith(".pdf") and url not in links:
            links.append(url)
    return links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "corpus" / "downloads")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    links = hr_pdf_links()
    print(f"Found {len(links)} PDF files.")
    for index, url in enumerate(links, start=1):
        name = clean_filename(Path(urlparse(url).path).name) or f"policy_{index}.pdf"
        path = args.out / name
        if path.exists():
            print(f"[skip] {name}")
            continue
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"[error] {url}: {error}")
            continue
        # A missing file often comes back as an HTML error page with a 200.
        if not response.content.startswith(b"%PDF"):
            print(f"[not a pdf] {url}")
            continue
        path.write_bytes(response.content)
        print(f"[{index}/{len(links)}] saved {path}")


if __name__ == "__main__":
    main()
