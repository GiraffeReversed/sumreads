#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import math
import json
import nltk
import matplotlib as mpl
import matplotlib.cm as cm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from summarizers import get_book_descriptions
from batch_processor import get_reviews_text

FILTER_SPOILERS = True
ASPECTS = ["book", "story", "writing", "characters", "pacing"]
SENT_THRESHOLD = 0.6
ENT_COUNT = 20


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

        print(key, len(descs))

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
    book_id = "64216"
    # harry potter and the sorcerer's stone = 3
    # twilight = 41865
    # the expanse = 8855321
    # eric = 64218
    # guards guards = 64216
    # eidolon = 25056040

    print("function started")
    text, title, series = get_reviews_text(book_id, FILTER_SPOILERS)
    print(title)
    print(series)

    print("chars", len(text))
    print("words", len(text.split()))

    result = get_book_descriptions(book_id, text, ENT_COUNT, ASPECTS, SENT_THRESHOLD, title, series)
    names_mapping, character_descriptions, aspect_descriptions, sentimental_descriptions, doc = result

    print(names_mapping)

    extra_stopwords = {
        "character", "characters", "writer", "author", "one", "first", "second", "last"
        "whole", "book", "books", "series", "man", "woman", "place", "next",
        "able", "read", "reader", "main", "many"
    }

    print_descriptions(character_descriptions, min_descs=10, extra_stopwords=extra_stopwords)
    print_descriptions(aspect_descriptions, min_descs=10, extra_stopwords=extra_stopwords)

    forbidden_sentimentals = {
        "thing", "things", "one", "way", "part", "deal", "job", "first",
        "book", "books", "serie", "series", "story", "stories", "read",
        "literature"
    }
    print_descriptions(sentimental_descriptions, min_descs=20, extra_stopwords=forbidden_sentimentals)

    character_desc_count = sum(len(word_descs[1]) for word_descs in character_descriptions.items())
    aspect_desc_count = sum(len(word_descs[1]) for word_descs in aspect_descriptions.items())
    sentimental_desc_count = sum(len(word_descs[1]) for word_descs in sentimental_descriptions.items())

    print("characters", character_desc_count)
    print("aspects", aspect_desc_count)
    print("sentimental", sentimental_desc_count)
    print("total", len(list(doc.sents)))
    print()

    sum_ = character_desc_count + aspect_desc_count + sentimental_desc_count
    total = len(list(doc.sents))

    print("percentage", f"{sum_ / total * 100:.2f} %")

    count = 0
    unique_descs = 0
    word_descs = 0
    for descss in [character_descriptions, aspect_descriptions, sentimental_descriptions]:
        for key, descs in descss.items():
            count = len(descs)
            unique_descs += len(set(" ".join(w.text for w in desc) for desc in descs))
            word_descs += sum(len(desc) for desc in descs)

    print("count", count)
    print("unique_descs", unique_descs)
    print("word_descs", word_descs)

    print(len(doc))
