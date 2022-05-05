#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
import en_core_web_md


def frequent_ngrams(words, n):
    stopwords = nltk.corpus.stopwords.words("english")
    ngrams = ngrams(words, n)
    filtered_ngrams = [
        ngram for ngram in ngrams
        if all(gram.lower() not in stopwords and gram.isalnum() for gram in ngram)
    ]
    return nltk.FreqDist(filtered_ngrams)


def _is_subseq(needle, haystack):
    return any(needle == haystack[i:i+len(needle)] for i in range(0, len(haystack)))


def _is_subset(needle, haystack):
    return set(needle).issubset(set(haystack))


def frequent_arbitrarygrams(words, min_, max_, count):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    words = [word.lower() for word in words if word.isalpha()]

    freq_ngrams = []
    for n in tqdm(range(max_, min_ - 1, -1)):
        ngramed = [ngram for ngram in ngrams(words, n) if n > 4 or not any(word in stopwords for word in ngram)]
        fd = nltk.FreqDist(ngramed)
        for ngram, ngram_freq in fd.most_common():
            for common, common_freq in freq_ngrams:
                if _is_subset(ngram, common):
                    fd.update({ngram: -ngram_freq})
        freq_ngrams.extend(fd.most_common(count))
    return freq_ngrams


def sentimental_words(lemmatized, total, treshold):
    sia = SentimentIntensityAnalyzer()
    fd = nltk.FreqDist(lemmatized)
    
    result = []
    for val, freq in fd.most_common():
        scores = sia.polarity_scores(val)
        if abs(scores["compound"]) > treshold:
            result.append((val, freq, scores))
        if len(result) >= total:
            break
    return result


def _no_unsquishables(needle, haystack, unsquishables):
    for h in haystack:
        if h in unsquishables and h not in needle:
            return False
    return True


def _is_word_aware_infix(needle, haystack, unsquishables):
    needle = re.split("[^a-zA-Z]+", needle.lower())
    haystack = re.split("[^a-zA-Z]+", haystack.lower())
    for i in range(len(haystack) - len(needle) + 1):
        if needle == haystack[i:i+len(needle)] and _no_unsquishables(needle, haystack, unsquishables):
            return True        
    return False


def _contained_in(needle_half, haystack, unsquishables):
    result = set()
    for hay in haystack:
        if needle_half != hay and _is_word_aware_infix(needle_half, hay, unsquishables):
            result.add(hay)
    return result


def _propagate(mapping):
    for from_, to_ in mapping.items():
        current = mapping[from_]
        stack = [from_]
        while current != mapping[to_]:
            stack.append(current)
            current = mapping[to_]
        for item in stack:
            mapping[item] = current
    return mapping


def squish(values, unsquishables=set(), unremapables=set()):
    unprocessed = set(values)
    mapping = {val:val for val in values}

    for val in values:
        if val not in unprocessed:
            continue
        for con in _contained_in(val, unprocessed, unsquishables):
            if not any(_is_word_aware_infix(con, word, unsquishables) for word in unremapables) or con.lower() == val.lower():
                mapping[con] = val
            unprocessed.remove(con)
        
    return _propagate(mapping)


def squish_ents(people, count, unremapables=set()):
    fdp = nltk.FreqDist(ent.text for ent in people)
    names = [p[0] for p in fdp.most_common(count)]
    return squish(names, unremapables=unremapables)


def compute_vocab_sentiment(doc):
    sia = SentimentIntensityAnalyzer()
    for word in doc.vocab:
        scores = sia.polarity_scores(word.text)
        doc.vocab[word.text].sentiment = scores["compound"]


def _get_dependent(token, dependencies, passthroughs=[], deep=False, result=None):
    result = result if result is not None else []
    for child in token.children:
        if child.dep_ in dependencies:
            result.append(child)
        if deep or child.dep_ in passthroughs:
            _get_dependent(child, dependencies, passthroughs, deep, result)
    
    return result


def _get_what(parent):
    what = []
    if not what:
        what = _get_dependent(parent, ["acomp"], ["xcomp"])
    if not what:
        what = _get_dependent(parent, ["attr"])
    if not what:
        what = _get_dependent(parent, ["pobj"], ["prep"])
    if not what or what[0].pos_ == "PRON":
        return []
    return what


def _get_negs(parent, what, old_negs=None):
    negs = []
    if not negs:
        negs = _get_dependent(parent, ["neg"])
    if not negs:
        negs = [det for det in _get_dependent(what, ["det"]) if det.text.strip().lower() == "no"]
    if not negs:
        return old_negs if old_negs is not None else []
    return negs


def _get_mods(what, start=None):
    preps = [what.head] if what.head.dep_ == "prep" and what.head.head == start else []
    mods = _get_dependent(
        what,
        ["amod", "compound", "advmod", "nummod", "npadvmod", "pobj", "prep", "poss", "case"],
        ["amod", "compound", "advmod", "nummod", "npadvmod", "pobj", "prep", "poss"],
    )
    mods = [mod for mod in mods if (start is None or mod.i > start.i) and mod.pos_ != "SCONJ"]
    return preps + mods


def _get_adj_mods(what, start=None):
    mods = _get_mods(what, start)
    return [mod for mod in mods if mod.pos_ in ["ADJ", "NOUN"] and mod.i < what.i]


def _get_description_from_mods(token):
    return [[mod] for mod in _get_adj_mods(token)]


def _get_description_from_aux(token, aux):
    whats = _get_what(aux)
    if not whats:
        return []

    result = []
    negs = _get_negs(aux, whats[0])
    while whats:
        if len(whats) > 1:
            result.append(whats)
            break
        
        what = whats[0]
        mods = _get_mods(what, aux)
        desc = sorted(negs + mods + [what], key=lambda tok: tok.i)
        result.append(desc)
    
        whats = _get_dependent(what, ["conj"])
    
    return result


def _get_descriptions(token):
    descs = _get_description_from_mods(token)
    if descs:
        return descs

    while token.dep_ == "conj":
        token = token.head

    if token.dep_ == "nsubj" and token.head.pos_ == "AUX":
        return _get_description_from_aux(token, token.head)
    return []


def _add_descriptions(i, word, descs, key, result, verbose):
    if not descs:
        return

    parent = word.head if word.head.pos_ == "AUX" else None
    if (verbose == 1 and not descs) or (verbose == 2 and descs) or verbose >= 3:
        print(i, word)
    if verbose >= 2:
        for desc in descs:
            print(" "*len(str(i)), word, parent if parent is not None else "is", *desc)
    if (verbose == 1 and not descs) or (verbose == 2 and descs) or verbose >= 3:
        print(word.sent)
        print()

    old_descs = result.get(key, [])
    old_descs.extend(descs)
    result[key] = old_descs


def get_character_descriptions(people, names_mapping, verbose=0):
    result = {}
    for i, person in enumerate(people):
        if person.text not in names_mapping:
            continue
        word = person.root
        descs = _get_descriptions(word)
        _add_descriptions(i, word, descs, names_mapping[person.text], result, verbose)
    
    return result


def get_aspect_descriptions(doc, aspects, verbose=0):
    result = {}
    for i, word in enumerate(doc):
        if word.text not in aspects:
            continue
        descs = _get_descriptions(word)
        _add_descriptions(i, word, descs, word.text, result, verbose)

    return result


def _get_descriptions_from_object(verb):
    whats = [what for what in _get_dependent(verb, ["dobj"]) if what.pos_ != "PRON" and what.text != verb.text]
    descs = []
    for what in whats:
        descs += [
            sorted(_get_negs(verb, what) + desc + [what], key=lambda word: word.i)
            for desc in _get_descriptions(what)
        ]
    return descs


def _get_described_entity_from_mod(mod):
    desc = []
    negs = []
    current = mod
    while current.dep_ in ["amod", "compound", "advmod", "attr"]:
        if current.head.lemma_.lower() != mod.lemma_.lower() and current.head.pos_ not in ["AUX"]:
            negs.extend(_get_dependent(current, ["neg"]))
            desc.append(current.head)
        current = current.head
    negs.extend(_get_dependent(current, ["neg"]))
    if not desc:
        return []
    return [sorted(negs + desc, key=lambda word:word.i)]


def _get_described_entity_from_subject(mod):
    if mod.dep_ == "nsubj":
        return []

    parent = mod.head
    while parent.text.strip() and parent.dep_ in ["amod", "compound", "advmod"]:
        parent = parent.head
    
    if parent.pos_ == "AUX":
        subjs = _get_dependent(parent, ["nsubj"])
        if len(subjs) == 1 and subjs[0].pos_ not in ("PROP", "PRON", "PUNCT", "SCONJ"):
            mods = _get_mods(subjs[0])
            mods.append(subjs[0])
            return [sorted(mods, key=lambda word: word.i)]
    return []


def get_sentimental_descriptions(doc, sent_threshold, verbose=0):
    result = {}
    for i, word in enumerate(doc):

        if abs(word.sentiment) < sent_threshold:
            continue

        key = word.text.lower()
        descs = []
        if word.pos_.startswith("N"):
            descs.extend(_get_descriptions(word))

        if word.pos_.startswith("V"):
            descs.extend(_get_descriptions_from_object(word))
            
        descs.extend(_get_described_entity_from_mod(word))
        descs.extend(_get_described_entity_from_subject(word))
        
        _add_descriptions(i, word, descs, key, result, verbose)

    return result


def get_book_descriptions(text, aspects, sent_threshold, title, series):
    nlp = en_core_web_md.load()
    nlp.max_length = len(text)

    doc = nlp(text)
    ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "WORK_OF_ART"]]

    names_mapping = squish_ents(ents, count=ent_count, unremapables={title, series})

    character_descriptions = get_character_descriptions(ents, names_mapping)
    aspect_descriptions = get_aspect_descriptions(doc, aspects)
    sentimental_descriptions = get_sentimental_descriptions(doc, sent_threshold)

    return names_mapping, character_descriptions, aspect_descriptions, sentimental_descriptions, doc
