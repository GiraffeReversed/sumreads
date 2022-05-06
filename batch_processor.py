import json
from tqdm import tqdm
import os
import shutil
from os.path import join, exists
from summarizers import get_book_descriptions
import pycld2 as cld2
import regex

RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")

SINGLE_REVIEWS_FOLDER = join("data", "singles")
SUMMARY_FOLDER = "summaries"
SPLIT_FOLDERS = 10 ** 3

FILTER_SPOILERS = True
ASPECTS = ["book", "story", "writing", "characters", "pacing"]
SENT_THRESHOLD = 0.6
ENT_COUNT = 20


# https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_reviews_path(book_id):
    hash_ = str(int(book_id) % SPLIT_FOLDERS)
    return join(SINGLE_REVIEWS_FOLDER, hash_, book_id)


def get_summary_path(book_id):
    hash_ = str(int(book_id) % SPLIT_FOLDERS)
    return join(SUMMARY_FOLDER, hash_, book_id)


def get_title_and_series(book):
    if book is None:
        return "", ""

    split = book["title"].split("(", 1)
    title = split[0].strip()
    series = book["title"].split("(", 1)[1].rstrip("0123456789#), ").strip() if len(split) > 1 else ""
    return title, series


# https://github.com/aboSamoor/polyglot/issues/71#issuecomment-707997790
def cleanup_utf8(text):
    return RE_BAD_CHARS.sub("", text)


def create_review_singles(reviews_filepath, books_filepath, recompute_all=True):
    reviews = {}
    pos = 0
    neg = 0
    with open(reviews_filepath) as f:
        for line in tqdm(f):
            review = json.loads(line)
            text = cleanup_utf8(review["review_text"])

            reliable, _, details = cld2.detect(text)
            if reliable:
                lang = details[0][1]
            else:
                lang = "en"

            if lang == "en":
                book_id = review["book_id"]
                this_book_data = reviews.get(book_id, {"reviews": []})
                this_book_data["reviews"].append(review)
                reviews[book_id] = this_book_data
                pos += 1
            else:
                neg += 1

    print(pos, neg)

    with open(books_filepath) as f:
        for line in tqdm(f):
            book = json.loads(line)
            book_id = book["book_id"]

            if book_id in reviews:
                title, series = get_title_and_series(book)
                reviews[book_id]["title"] = title
                reviews[book_id]["series"] = series

    for book_id, this_book_data in tqdm(reviews.items()):
        if not book_id.isdecimal():
            print("WEIRD BOOK_ID", book_id)
            continue

        singles_file_path = get_reviews_path(book_id)
        if not exists(singles_file_path) or recompute_all:
            with open(singles_file_path, "w") as f:
                f.write(json.dumps(this_book_data))


def filter_spoilerous(reviews):
    return [review for review in reviews if "spoiler" not in review["review_text"]]


def find_book(book_id):
    with open("data/goodreads_books_fantasy_paranormal.json") as f:
        for line in tqdm(f):
            book = json.loads(line)
            if book["book_id"] == book_id:
                return book
    return None


def get_reviews_from_book_id(book_id, genre):
    book_singles_path = get_reviews_path(book_id)
    if os.path.exists(book_singles_path):
        with open(book_singles_path) as f:
            reviews = json.loads(f.read())
            return reviews["reviews"], reviews["title"], reviews["series"]

    if genre is None:
        return [], "", ""

    reviews = []
    with open(get_filename("reviews", genre)) as f:
        for line in tqdm(f):
            review = json.loads(line)
            if review["book_id"] == book_id:
                reviews.append(review)

    book = find_book(book_id)
    title, series = get_title_and_series(book)
    return reviews, title, series


def get_reviews_text(book_id, no_spoilerous, genre):
    reviews, title, series = get_reviews_from_book_id(book_id, genre)

    if no_spoilerous:
        reviews = filter_spoilerous(reviews)

    return " ".join(review["review_text"] for review in reviews), title, series


def token_desc_to_str(desc):
    return " ".join(w.text for w in desc).lower().replace(" '", "'").replace("n't", "not")


def serialize_descriptions(descss):
    return {
        key: [token_desc_to_str(desc) for desc in descs]
        for key, descs in descss.items()
    }


def create_summary(book_id, force_rebuild=True):
    summary_path = get_summary_path(book_id)
    if exists(summary_path) and not force_rebuild:
        return

    text, title, series = get_reviews_text(book_id, FILTER_SPOILERS, None)

    result = get_book_descriptions(
        book_id, text, ENT_COUNT, ASPECTS, SENT_THRESHOLD, title, series)
    names_mapping, character_descriptions, aspect_descriptions, sentimental_descriptions, doc = result

    with open(summary_path, "w") as f:
        f.write(json.dumps({
            "book_id": book_id,
            "title": title,
            "series": series,
            "names_mapping": names_mapping,
            "characters": serialize_descriptions(character_descriptions),
            "aspects": serialize_descriptions(aspect_descriptions),
            "sentimental": serialize_descriptions(sentimental_descriptions),
            "words_count": len(doc),
            "sents_count": len(list(doc.sents))
        }))


def get_book_ids():
    result = []
    for foldername in os.listdir(SINGLE_REVIEWS_FOLDER):
        folder_path = join(SINGLE_REVIEWS_FOLDER, foldername)
        assert os.path.isdir(folder_path)
        result.extend(os.listdir(folder_path))
    return result


def create_summaries(from_=0, to_=2*10**6):
    for book_id in tqdm(get_book_ids()[from_:to_]):
        create_summary(book_id, force_rebuild=False)


def get_filename(type_, genre):
    return f"data/goodreads_{type_}_{genre}.json"


def prepare_split_folders(folder):
    for i in range(SPLIT_FOLDERS):
        os.makedirs(join(folder, str(i)), exist_ok=True)


if __name__ == "__main__":
    # delete_folder_contents(SINGLE_REVIEWS_FOLDER)
    # print("reviews deleted...")

    # prepare_split_folders(SINGLE_REVIEWS_FOLDER)
    # print("review folders created...")

    # for genre in ["fantasy_paranormal", "young_adult"][1:]:
    #     create_review_singles(get_filename("reviews", genre), get_filename("books", genre))
    #     print(genre, "files created...")

    # delete_folder_contents(SUMMARY_FOLDER)
    # print("summaries deleted...")

    # prepare_split_folders(SUMMARY_FOLDER)
    # print("summary folders created...")

    create_summaries(150000, 200000)
    pass
