"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import pickle
import re
import string
import unittest
# import nltk for downloading the list of stopwords
import nltk
from collections import namedtuple
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path
from time import perf_counter
from typing import Generator, Dict, Optional, Union

import numpy as np
from nltk.stem import *  # Allowed stemming library
from nltk.corpus import stopwords  # Allowed stopwords list

APOSTROPHES_REGEX = r'[\u0022\u0027\u0060\u00AB\u00B4\u00BB\u2018\u2019\u201B\u201C\u201D\u201E\u201F\u2039\u203A' \
                    r'\u275B\u275C\u275D\u275E\u275F\u276E\u276F]'  # Unicode apostrophes
# HYPHEN_REGEX = r'(?<=\d)[-–—―](?<=\d)'
HYPHEN_REGEX = r'[\u002D\u058A\u05BE\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40' \
               r'\u2E5D\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D\u10EAD' \
               r'\u005F\u203F\u2040\u2054\uFE33\uFE34\uFE4D\uFE4E\uFE4F\uFF3F]'  # # Unicode 'Punctuation, Dash' and 'Punctuation, Connector'
PUNCTUATION_REGEX = r'(?<!\d)[^\w\s](?!\d)'

Article = namedtuple("Article", ["title", "title_id", "bdy"])
stemmer = SnowballStemmer("english", ignore_stopwords=False)

stemmer_cache = {}

nltk.download('stopwords')
stop_words = stopwords.words('english')


# TODO maybe ignore stopwords to improve performance


class ArticlesParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles: list[Article] = []
        self._title: Optional[str] = None
        self._title_id: Optional[int] = None
        self._bdy: Optional[str] = None
        self._previous_tag, self._current_tag = None, None
        self._previous_data, self._current_data = None, None

    def handle_starttag(self, tag, attrs):
        self._previous_tag, self._current_tag = self._current_tag, tag
        self._previous_data, self._current_data = self._current_data, ''

    def handle_endtag(self, tag):
        if tag == 'title':
            self._title = self._current_data
        elif tag == 'bdy':
            self._bdy = self._current_data
        elif tag == 'id' and self._previous_tag == 'title':
            self._title_id = int(self._current_data)
        elif tag == 'article':
            self.articles.append(Article(self._title, self._title_id, self._bdy))

    def handle_data(self, data):
        if self._current_tag in ['title', 'bdy', 'id']:
            self._current_data += data


class InvertedIndex:
    dump_file = Path('../index.obj')

    def __init__(self):
        self.tokens: Dict[str, Union[list, np.array]] = {}
        self.articles: Dict[int, Path] = {}

    def populate(self, path: Path, articles_total: int = 281782):
        documents = path.iterdir()
        # __benchmark__ {
        articles_processed = 0
        start = perf_counter()
        # __benchmark__ }
        for document in documents:
            for article in get_articles(document):
                for token in text2tokens(article.bdy):
                    if token in self.tokens:
                        self.tokens[token].append(article.title_id)
                    else:
                        self.tokens[token] = [article.title_id]
                self.articles[article.title_id] = document
                if articles_processed == articles_total:
                    break
                # __benchmark__ {
                articles_processed += 1
                articles_per_second = articles_processed / (perf_counter() - start)
                seconds_remaining = (articles_total - articles_processed) / articles_per_second
                print(f'\rIndexing: {articles_processed}/{articles_total} ({articles_per_second:.2f} articles/s)',
                      f'[Estimated remaining time: {timedelta(seconds=seconds_remaining)}]', end='')
                # __benchmark__ }
            if articles_processed == articles_total:
                break
        # __benchmark__ {
        print()
        articles_per_second = articles_processed / (perf_counter() - start)
        print(f'Estimated total indexing time: {timedelta(seconds=281782 / articles_per_second)}')
        # __benchmark__ }
        self._optimise()

    def _optimise(self):
        # __benchmark__ {
        start = perf_counter()
        # __benchmark__ }
        for token in self.tokens:
            self.tokens[token] = np.array(self.tokens[token])
        # __benchmark__ {
        print(f'Optimise tokens index: {timedelta(seconds=perf_counter() - start)}')
        # __benchmark__ }

    def fetch(self, article_title_id: int) -> Optional[Article]:
        document = self.articles[article_title_id]
        for article in get_articles(document):
            if article.title_id == article_title_id:
                return article

    def search(self, term: str) -> Optional[Union[list, np.array]]:
        return self.tokens.get(term)

    def dump(self):
        with InvertedIndex.dump_file.open('wb') as file:
            pickle.dump(self, file)
        print(f'Bytes written: {InvertedIndex.dump_file.stat().st_size:}')

    @staticmethod
    def load() -> 'InvertedIndex':
        with InvertedIndex.dump_file.open('rb') as file:
            return pickle.load(file)

    def __str__(self):
        return str(self.tokens)


def text2tokens(text):
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """
    # TODO: tokenization
    # TODO: lowercasing
    # TODO: stemming (library allowed, english only, e. g. NLTK)
    # TODO: remove stop words
    # TODO: handle special characters, multi-blanks, punctuation
    # TODO: test preprocessing (here points are given)!

    tokens = tokenize(text)
    tokens = lowercase(tokens)
    tokens = stem(tokens)
    tokens = remove_stop_words(tokens)

    return list(tokens)


def tokenize(text):
    """
    :param text: input text
    :return: single tokens found in the input text
    """
    # First split text at whitespaces since we do not build a biword index
    tokens = split_at_whitespaces(text)
    # Remove everything after apostrophes (including the apostrophe itself)
    tokens = remove_apostrophes(tokens)
    # Split at hyphens to safe hyphenated sequence as two tokens
    tokens = remove_hyphen(tokens)
    # Remove any punctuations
    tokens = remove_punctuation(tokens)

    yield from tokens


def split_at_whitespaces(text):
    for word in text.split():
        yield word


def remove_apostrophes(tokens):
    for token in tokens:
        yield re.split(APOSTROPHES_REGEX, token)[0]


def remove_hyphen(tokens):
    for token in tokens:
        for portion in re.split(HYPHEN_REGEX, token):
            yield portion


def remove_punctuation(tokens):
    # remove punctuation only from words not for numbers
    # (?<!...) is called negative lookbehind assertion https://docs.python.org/3/library/re.html
    for token in tokens:
        token = re.sub(PUNCTUATION_REGEX, '', token)
        token = token.strip()
        if token:  # matches empty string ''
            yield token


def lowercase(tokens):
    for token in tokens:
        yield token.lower()


def stem(tokens):
    """
    This method uses the NLTK Porter stemmer
    :param tokens: A string ready for stemming
    :return: The stemmed string
    """
    for token in tokens:
        if token in stemmer_cache:
            yield stemmer_cache[token]
        else:
            stemmed_token = stemmer.stem(token)
            stemmer_cache[token] = stemmed_token
            yield stemmed_token


def remove_stop_words(tokens):
    for token in tokens:
        if token not in stop_words:
            yield token


def get_articles(document: Path) -> Generator[Article, None, None]:
    parser = ArticlesParser()
    with document.open(encoding='utf-8') as file:
        for line in file:
            parser.feed(line)
        yield from parser.articles
        parser.articles.clear()


class TestTextPreProcessing(unittest.TestCase):

    def test_remove_apostrophes(self):
        input_text = ['nation\'s', 'Australia\'s', 'other\'s', 'another`s', 'different´s', 'apostrophes', 'Cabezón\'\'']
        expected_text = ['nation', 'Australia', 'other', 'another', 'different', 'apostrophes', 'Cabezón']
        self.assertEqual(list(remove_apostrophes(input_text)), expected_text)

    def test_remove_hyphen(self):
        input_text = ['simple-hyphen', 'ndash–hyphen', 'mdash—hyphen', ' horbar―hyphen', '0-679-73392-2',
                      '  1805–1872', ' (1856–1953)']
        expected_text = ['simple', 'hyphen', 'ndash', 'hyphen', 'mdash', 'hyphen', ' horbar', 'hyphen',
                         '0', '679', '73392', '2', '  1805', '1872', ' (1856', '1953)']
        self.assertEqual(list(remove_hyphen(input_text)), expected_text)

    def test_remove_punctuation(self):
        input_text = ['Apostrophes:', ' , ', ' men. ', 's,', '(commanding),', '2.74', '(((.corps)', 's).']
        expected_text = ['Apostrophes', 'men', 's', 'commanding', '2.74', 'corps', 's']
        self.assertEqual(list(remove_punctuation(input_text)), expected_text)

    def test_stemming(self):
        input_text = ['caresses', 'flies', 'flies', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned',
                      'humbled', 'sized',
                      'meeting', 'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference',
                      'colonizer', 'plotted']
        expected_text = ['caress', 'fli', 'fli', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own', 'humbl', 'size',
                         'meet', 'state', 'siez', 'item', 'sensat', 'tradit', 'refer', 'colon', 'plot']
        self.assertEqual(list(stem(input_text)), expected_text)

    def test_remove_stop_words(self):
        input_text = ['i', 'me', 'my', 'notremoved', 'myself', 'we', 'stays', 'our', 'you', 'nope', "you're", "you've", 'stillhere']
        expected_text = ['notremoved', 'stays', 'nope', 'stillhere']
        self.assertEqual(list(remove_stop_words(input_text)), expected_text)


def main():
    index = InvertedIndex()
    index.populate(Path('../dataset/wikipedia articles'), articles_total=200)
    index.dump()


if __name__ == "__main__":
    main()
