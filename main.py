#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import json
import math
import spacy
from spacy import displacy
import en_core_web_md
import matplotlib as mpl
import matplotlib.cm as cm
from nltk.stem import WordNetLemmatizer
import nltk

from summarizers import frequent_arbitrarygrams
from summarizers import sentimental_words
from summarizers import squish_ents
from summarizers import compute_vocab_sentiment
from summarizers import get_character_descriptions
from summarizers import get_aspect_descriptions
from summarizers import get_sentimental_descriptions


FILTER_SPOILERS = True


def find_book(book_id):
    with open("data/goodreads_books_fantasy_paranormal.json") as f:
        for line in tqdm(f):
            book = json.loads(line)
            if book["book_id"] == book_id:
                return book
    return None


def get_book_and_series(book_id):
    book = find_book(book_id)
    if book is None:
        return "", ""

    title = book["title"].split("(", 1)[0]
    series = book["title"].split("(", 1)[1].rstrip("0123456789#), ")
    return title if title else "", series if series else ""


def get_reviews_from_book_id(book_id):
    reviews = []
    with open("data/goodreads_reviews_fantasy_paranormal.json") as f:
        for line in tqdm(f):
            review = json.loads(line)
            if review["book_id"] == book_id:
                reviews.append(review)
    return reviews


def filter_spoilerous(reviews):
    return [review for review in reviews if "spoiler" not in review["review_text"]]


def get_ids():
    ids = set()
    with open("data/goodreads_reviews_fantasy_paranormal.json") as f:
        for line in tqdm(f):
            review = json.loads(line)
            ids.add(review["book_id"])
    return ids


def word_to_color(word, *args, **kwargs):
    sentiment = doc.vocab[word].sentiment

    norm = mpl.colors.Normalize(vmin=-0.8, vmax=0.8)
    cmap = cm.RdYlGn

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    color = tuple(map(lambda v: int(255*v), m.to_rgba(sentiment)[:3]))
    return color if color != (254, 254, 189) else (200, 200, 160)


def print_descriptions(descs, min_descs=15, extra_stopwords=set()):
    forbidden = set(nltk.corpus.stopwords.words("english")) | extra_stopwords | descs.keys()
    for key, descs in sorted(descs.items(), key=lambda pair: len(pair[1]), reverse=True):
        descs = [" ".join(word.text if word.text != "n't" else "not" for word in desc).lower() for desc in descs]
        if len(descs) < min_descs:
            continue

        print(key, len(descs), end=" ")

        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    
        wordcloud = WordCloud(
            width=1200,
            height=600,
            stopwords=forbidden,
            collocation_threshold=10,
            color_func=word_to_color
        ).generate(" ".join(descs))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
        print(*sorted(set(descs), key=len, reverse=True)[:int(math.log(len(descs), 2))], sep="\n")
        print()


if __name__ == "__main__":
    book_id = "41865"
    # harry potter and the sorcerer's stone = 3
    # twilight = 41865
    # the expanse = 8855321
    # eric = 64218
    # guards guards = 64216

    title, series = get_book_and_series(book_id)
    print(title)
    print(series)

    reviews = get_reviews_from_book_id(book_id)
    print(len(reviews))

    if FILTER_SPOILERS:
        reviews = filter_spoilerous(reviews)
        print(len(reviews))

    text = " ".join(review["review_text"] for review in reviews)

    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, pos.lower()[0] if pos.lower()[0] in "nvars" else "n").lower()
        for word, pos in nltk.pos_tag(nltk.word_tokenize(text))
    ]

    print(frequent_arbitrarygrams(lemmatized, 2, 6, 3))
    print(sentimental_words(lemmatized, 20, 0.5))

    nlp = en_core_web_md.load()
    nlp.max_length = len(text)


    print("chars", len(text))
    print("words", len(text.split()))


    doc = nlp(text)
    ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "WORK_OF_ART"]]


    names_mapping = squish_ents(ents, unremapables={title, series})
    print(names_mapping)

    compute_vocab_sentiment(doc)

    character_descriptions = get_character_descriptions(ents, names_mapping)
    aspect_descriptions = get_aspect_descriptions(doc, ["book", "story", "writing", "characters", "pacing"])

    extra_stopwords = {
        "character", "characters", "writer", "author", "one", "first", "second", "last"
        "whole", "book", "books", "series", "man", "woman", "place", "next",
        "able", "read", "reader", "main", "many"
    }

    print_descriptions(character_descriptions, min_descs=10, extra_stopwords=extra_stopwords)
    print_descriptions(aspect_descriptions, min_descs=10, extra_stopwords=extra_stopwords)

    sentimental_descriptions = get_sentimental_descriptions(doc, 0.6)

    forbidden_sentimentals = {
        "thing", "things", "one", "way", "part", "deal", "job", "first",
        "book", "books", "serie", "series", "story", "stories", "read",
        "literature"
    }
    print_descriptions(sentimental_descriptions, min_descs=10, extra_stopwords=forbidden_sentimentals)


    print("characters", sum(len(foo[1]) for foo in character_descriptions.items()))
    print("aspects", sum(len(foo[1]) for foo in aspect_descriptions.items()))
    print("sentimental", sum(len(foo[1]) for foo in sentimental_descriptions.items()))
    print("total", len(list(doc.sents)))
    print()

    sum_ = sum(len(foo[1]) for foo in character_descriptions.items()) + sum(len(foo[1]) for foo in aspect_descriptions.items()) + sum(len(foo[1]) for foo in sentimental_descriptions.items())

    total = len(list(doc.sents))

    print("percentage", f"{sum_ / total * 100:.2f} %")