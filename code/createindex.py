"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization
function (text2tokens), there are no restrictions in how you organize this file.
"""
from __future__ import annotations

import array
import pickle
import re
import unittest
from collections import namedtuple, Counter, defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import timedelta
from functools import cached_property, partial, lru_cache
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path
from time import perf_counter
from typing import Generator, Optional

from nltk.corpus import stopwords  # Allowed stopwords list
from nltk.stem import *  # Allowed stemming library

APOSTROPHES_REGEX = r'[\u0060\u00B4\2018\u2019\u0027\u2032\u02BB]'  # Unicode apostrophes

# HYPHEN_REGEX = r'(?<=\d)[-–—―](?<=\d)'
HYPHEN_REGEX = r'[\u002D\u058A\u05BE\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40' \
               r'\u2E5D\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D\u10EAD' \
               r'\u005F\u203F\u2040\u2054\uFE33\uFE34\uFE4D\uFE4E\uFE4F\uFF3F]'  # Unicode 'Punctuation, Dash'

# 'Punctuation, Connector'
# (?<!...) is called negative lookbehind assertion https://docs.python.org/3/library/re.html
PUNCTUATION_REGEX = r'(?<!\d)[^\w\s](?!\d)'

stemmer = SnowballStemmer("english", ignore_stopwords=False)
stop_words = set(stopwords.words('english'))  # set allows O(1) lookup, list is O(n)

Article = namedtuple("Article", ["title", "title_id", "bdy"])
Posting = namedtuple("Posting", ["article_title_id", "article_len", "tf"])
I_array = partial(array.array, 'I')


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
        self._previous_data, self._current_data = self._current_data, []

    def handle_endtag(self, tag):
        if tag == 'title':
            self._title = ''.join(self._current_data)
        elif tag == 'bdy':
            self._bdy = ''.join(self._current_data)
        elif tag == 'id' and self._previous_tag == 'title':
            self._title_id = int(''.join(self._current_data))
        elif tag == 'article':
            self.articles.append(Article(self._title, self._title_id, self._bdy))

    def handle_data(self, data):
        if self._current_tag in ['title', 'bdy', 'id']:
            self._current_data.append(data)


class InvertedIndex:
    def __init__(self):
        self.article_count: int = 0
        self._total_article_len: int = 0
        # Note: We use array.array instead of np.array, because array is over-allocated and thus allows efficient
        # .append(). (https://github.com/python/cpython/blob/main/Modules/arraymodule.c#L153)
        self._tokens: dict[str, array.array] = defaultdict(I_array)
        self._articles: dict[int, array.array] = defaultdict(I_array)
        self._path: Optional[Path] = None

    @cached_property
    def avg_article_len(self):
        return self._total_article_len / self.article_count

    def populate(self, path: str, articles_total: int = 281782):
        self._path = Path(path)
        documents = self._path.iterdir()
        # __benchmark__ {
        start = perf_counter()
        # __benchmark__ }
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(get_tokens_for_document, document): document for document in documents}
            for future in as_completed(futures):
                tokens_for_document = future.result()
                for article_title_id, tokens in tokens_for_document.items():
                    for token, tf in tokens.items():
                        self._tokens[token].append(article_title_id)
                        self._tokens[token].append(tf)
                    self.article_count += 1
                    document = futures[future]
                    # maybe len(article.bdy) instead of tokens count
                    article_len = sum(tokens.values())
                    self._articles[article_title_id].append(int(document.stem))
                    self._articles[article_title_id].append(article_len)
                    self._total_article_len += article_len
                # frees processed futures from memory (important!)
                del futures[future]
                # __benchmark__ {
                articles_per_second = self.article_count / (perf_counter() - start)
                seconds_remaining = (articles_total - self.article_count) / articles_per_second
                print(f'\rIndexing: {self.article_count}/{articles_total} ({articles_per_second:.2f} articles/s)',
                      f'[Estimated remaining time: {timedelta(seconds=seconds_remaining)}]', end='')
                # __benchmark__ }
        # __benchmark__ {
        print(f'\nTotal indexing time: {timedelta(seconds=perf_counter() - start)}')
        # __benchmark__ }

    def fetch(self, article_title_id: int) -> Article:
        document_stem, _ = self._articles[article_title_id]
        document = Path(self._path, f'{document_stem}.xml')
        for article in get_articles(document):
            if article.title_id == article_title_id:
                return article

    def search(self, term: str) -> Generator[Posting, None, None]:
        for article_title_id, tf in grouper(self._tokens[term], 2):
            _, article_len = self._articles[article_title_id]
            yield Posting(article_title_id, article_len, tf)

    def dump(self, path: str):
        path = Path(path)
        with path.open('wb') as file:
            pickle.dump(self, file)
        print(f'Bytes written: {path.stat().st_size:}')

    @staticmethod
    def load(path: str) -> 'InvertedIndex':
        path = Path(path)
        with path.open('rb') as file:
            return pickle.load(file)

    def __str__(self):
        return str(self._tokens)


def text2tokens(text):
    """
    :param text: a text string
    :return: a tokenized string with preprocessing (e.g. stemming, stopword removal, ...) applied
    """

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
    This method uses the NLTK Snowball stemmer
    :param tokens: A string ready for stemming
    :return: The stemmed string
    """
    for token in tokens:
        yield lookup_stem(token)


@lru_cache(maxsize=150000)
def lookup_stem(token):
    return stemmer.stem(token)


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


def get_tokens_for_document(document: Path) -> dict[int, Counter]:
    d = {}
    for article in get_articles(document):
        d[article.title_id] = Counter(text2tokens(article.bdy))
        d[article.title_id].update(text2tokens(article.title))
    return d


def grouper(iterable, n, fill_value=None):
    """
    Collect data into non-overlapping fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    # (https://docs.python.org/3/library/itertools.html#itertools-recipes)
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


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
        input_text = ['davis', 'caresses', 'flies', 'flies', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed',
                      'owned',
                      'humbled', 'sized',
                      'meeting', 'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference',
                      'colonizer', 'plotted']
        expected_text = ['davi', 'caress', 'fli', 'fli', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own', 'humbl',
                         'size',
                         'meet', 'state', 'siez', 'item', 'sensat', 'tradit', 'refer', 'colon', 'plot']
        self.assertEqual(list(stem(input_text)), expected_text)

    def test_remove_stop_words(self):
        input_text = ['this', 'is', 'a', 'i', 'me', 'my', 'notremoved', 'myself', 'we', 'stays', 'our', 'you', 'nope',
                      "you're", "you've",
                      'stillhere']
        expected_text = ['notremoved', 'stays', 'nope', 'stillhere']
        self.assertEqual(list(remove_stop_words(input_text)), expected_text)

    def test_text2tokens(self):
        input_text = 'This     "is" a. e@xample "Sentence"   "related words"    2   .te.s.t. 12345 .our. P.reprOceSSing'
        expected_text = ['exampl', 'sentenc', 'relat', 'word', '2', 'test', '12345', 'preprocess']
        self.assertEqual(text2tokens(input_text), expected_text)
        self.assertEqual(text2tokens('mother\'s day'), ['mother', 'day'])
        self.assertEqual(text2tokens('Computer "Operating Systems"'), ['comput', 'oper', 'system'])
        self.assertEqual(text2tokens('"tai chi" styles forms'), ['tai', 'chi', 'style', 'form'])
        self.assertEqual(text2tokens('"Apple Inc" products invented by "Steve Jobs"'),
                         ['appl', 'inc', 'product', 'invent', 'steve', 'job'])
        self.assertEqual(
            text2tokens('Jazz "Charles Mingus" "Miles Davis" collaboration interaction personal relationship -album'),
            ['jazz', 'charl', 'mingus', 'mile', 'avi', 'collabor', 'interact', 'person', 'relationship', 'album'])
        self.assertEqual(text2tokens('predictive analysis +logistic +regression model program application'),
                         ['predict', 'analysi', 'logist', 'regress', 'model', 'program', 'applic'])


def main():
    index = InvertedIndex()
    index.populate('../dataset/wikipedia articles')
    index.dump('../index.obj')


if __name__ == "__main__":
    main()
