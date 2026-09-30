
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www.kgh.nhs.uk/policies-and-procedures/"
OUTPUT_DIR = "corpus/employer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}


def clean_filename(filename):
    """Remove invalid characters from filenames."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename).strip()


def download_hr_pdfs():
    response = requests.get(BASE_URL, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Find the Human Resources section heading
    hr_heading = None

    for heading in soup.find_all(["h2", "h3"]):
        text = heading.get_text(" ", strip=True).lower()

        if "human resources policies and procedures" in text:
            hr_heading = heading
            break

    if not hr_heading:
        raise RuntimeError(
            "Human Resources section not found. "
            "The website structure may have changed."
        )

    pdf_links = []

    # Collect links until the next section heading
    for element in hr_heading.find_all_next():

        if element.name in ["h2", "h3"] and element != hr_heading:
            break

        if element.name != "a":
            continue

        href = element.get("href")

        if not href:
            continue

        url = urljoin(BASE_URL, href)

        # Only collect PDF links
        if urlparse(url).path.lower().endswith(".pdf"):
            if url not in pdf_links:
                pdf_links.append(url)

    print(f"Found {len(pdf_links)} PDF files.")

    for index, pdf_url in enumerate(pdf_links, start=1):

        filename = os.path.basename(urlparse(pdf_url).path)

        if not filename.lower().endswith(".pdf"):
            filename = f"policy_{index}.pdf"

        filename = clean_filename(filename)

        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            print(f"[SKIP] {filename}")
            continue

        try:
            print(f"[{index}/{len(pdf_links)}] Downloading {filename}")

            pdf_response = requests.get(
                pdf_url,
                headers=headers,
                timeout=60
            )

            pdf_response.raise_for_status()

            # Verify the response appears to be a PDF
            if not pdf_response.content.startswith(b"%PDF"):
                print(f"[WARNING] Not a valid PDF: {pdf_url}")
                continue

            with open(filepath, "wb") as file:
                file.write(pdf_response.content)

            print(f"[SAVED] {filepath}")

        except requests.RequestException as error:
            print(f"[ERROR] {pdf_url}: {error}")

    print("\nDownload completed!")


if __name__ == "__main__":
    download_hr_pdfs()