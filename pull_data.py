import csv
import requests
from io import StringIO
from io import BytesIO
from pathlib import Path
import logging
import zipfile
import re

GUTENBERG_CSV_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"

r = requests.get(GUTENBERG_CSV_URL)
csv_text = r.content.decode("utf-8")

plato_books = [book for book in csv.DictReader(StringIO(csv_text))
                   if 'Aristotle' in book['Authors'] and book['Language']=='en']



GUTENBERG_ROBOT_URL = "http://www.gutenberg.org/robot/harvest?filetypes[]=txt"
r = requests.get(GUTENBERG_ROBOT_URL)


GUTENBERG_MIRROR = re.search('(https?://[^/]+)[^"]*.zip', r.text).group(1)
GUTENBERG_TEXT_URL = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
GUTENBERG_TEXT = "PROJECT GUTENBERG EBOOK "

def gutenberg_text_urls(id: str, mirror=GUTENBERG_MIRROR, suffixes=("", "-8", "-0")) -> list[str]:
    path = "/".join(id[:-1]) or "0"
    return [f"{mirror}/{path}/{id}/{id}{suffix}.zip" for suffix in suffixes]

# book_id = plato_books[0]["Text#"]
# gutenberg_text_urls(book_id)


def download_gutenberg(id: str) -> str:
    for url in gutenberg_text_urls(id):
        r = requests.get(url)
        if r.status_code == 404:
            logging.warning(f"404 for {url}")
            continue
        r.raise_for_status()
        break

    z = zipfile.ZipFile(BytesIO(r.content))

    if len(z.namelist()) != 1:
        raise Exception(f"Expected 1 file in {z.namelist()}")

    return z.read(z.namelist()[0]).decode('utf-8')

# text = download_gutenberg(book_id)
#
# print(text[:1500])



# lines = text.splitlines()
#
# first = True
# for idx, line in enumerate(lines):
#     if GUTENBERG_TEXT in line:
#         if first:
#             first = False
#             continue
#         print('=' * 80)
#         print('\n'.join(lines[idx-20:idx+20]))
#         print('=' * 80)
#         print()


def strip_headers(text):
    in_text = False
    output = []

    for line in text.splitlines():
        if GUTENBERG_TEXT in line:
            if not in_text:
                in_text = True
            else:
                break
        else:
            if in_text:
                output.append(line)

    return "\n".join(output).strip()


# stripped_text = strip_headers(text)

def book_text(book_id):
    r = requests.get(GUTENBERG_TEXT_URL.format(id=book_id))
    text = r.text
    clean_text = strip_headers(text)
    return clean_text

data_path = Path("data_aristotle")
data_path.mkdir(exist_ok=True)

for book in plato_books:
    id = book["Text#"]
    text = book_text(id)
    print(f"Saving {book['Title']} by {book['Authors']} containing {len(text):_} characters")
    with open(data_path / (id + ".txt"), "wt") as f:
        f.write(text)

with open(data_path / 'metadata.csv', 'wt') as f:
    csv_writer = csv.DictWriter(f, fieldnames=plato_books[0].keys())
    csv_writer.writeheader()
    for book in plato_books:
        csv_writer.writerow(book)