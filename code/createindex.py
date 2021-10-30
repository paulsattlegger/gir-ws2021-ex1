"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import pickle
import re
import string
from collections import namedtuple
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Generator, Dict, List, Optional

import numpy as np
from nltk.stem import *

APOSTROPHES_REGEX = r'[\'`´]'
HYPHEN_REGEX = r'(?<=\d)[-–—―](?<=\d)'
PUNCTUATION_REGEX = r'(?<!\d)[^\w\s](?!\d)'

Article = namedtuple("Article", ["title", "title_id", "bdy"])
stemmer = PorterStemmer()


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
    def __init__(self, base_dir: str):
        self.data: Dict[str, List[np.array]] = {}
        self.article_count = 0
        self._base_dir = base_dir

    def insert(self, term: str, posting: np.array):
        if term in self.data:
            self.data[term].append(posting)
        else:
            self.data[term] = [posting]

    def index(self):
        documents = Path(self._base_dir).iterdir()
        # TODO: remove from final version
        documents = islice(documents, 5)
        for document in documents:
            for article in get_articles(document):
                self.article_count += 1
                for token in text2tokens(article.bdy):
                    posting = np.array([document.stem, article.title_id], dtype=np.uint32)
                    self.insert(token, posting)

    def fetch(self, posting: np.array) -> Optional[Article]:
        document_id, article_title_id = posting
        document = Path(self._base_dir, f'{document_id}.xml')
        for article in get_articles(document):
            if article.title_id == article_title_id:
                return article

    def search(self, term: str) -> Optional[list[np.array]]:
        return self.data.get(term)

    def dump(self):
        with open('../index.obj', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load() -> 'InvertedIndex':
        with open('../index.obj', 'rb') as file:
            return pickle.load(file)

    def __str__(self):
        return str(self.data)


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
    for token in tokenize(text):
        token = token.lower()
        yield stem(token)


def tokenize(text):
    """
    :param text: input text
    :return: single tokens seperated by whitespace
    """
    for word in text.split():
        word = remove_apostrophes(word)
        word = remove_punctuation(word)
        if word != "":  # remove "empty" words (if after removal nothing is left)
            yield from remove_hyphen(word)


def remove_apostrophes(token):
    # return token.split("'")[0]
    return re.split(APOSTROPHES_REGEX, token)[0]


def remove_hyphen(token):
    # remove hyphen only for words but not for numbers
    # (?<!...) is called negative lookbehind assertion https://docs.python.org/3/library/re.html
    for word in re.split(HYPHEN_REGEX, token):
        yield word


def remove_punctuation(token):
    # return re.split('\.|,|;', token)[0]
    # remove punctuation only from words not for numbers
    # (?<!...) is called negative lookbehind assertion https://docs.python.org/3/library/re.html
    token = re.sub(PUNCTUATION_REGEX, '', token)
    return token.strip(string.punctuation)  # to cover (1856–1953)


def stem(term):
    """
    This method uses the NLTK Porter stemmer
    :param term: A term string ready for stemming
    :return: the stemmed string
    """
    return stemmer.stem(term)


def get_articles(document: Path) -> Generator[Article, None, None]:
    parser = ArticlesParser()
    with open(document, encoding='utf-8') as file:
        for line in file:
            parser.feed(line)
        yield from parser.articles
        parser.articles.clear()


def get_stats(articles_processed: int, start: float, articles_total: int = 281782):
    # $ cat * | grep "<article" | wc -l == 281782
    end = perf_counter()
    articles_per_second = articles_processed / (end - start)
    print(f'Performance: {articles_per_second:.2f} articles/s')
    print(f'Estimated total indexing time: {articles_total / articles_per_second / 60:.2f} m')


def main():
    index = InvertedIndex('../dataset/wikipedia articles')
    start = perf_counter()
    index.index()
    get_stats(index.article_count, start)
    index.dump()
    return

    """plurals = ['caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned', 'humbled', 'sized', 'meeting',
               'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference', 'colonizer', 'plotted']

    singles = [stem(plural) for plural in plurals]
    print(' '.join(singles))"""

    test_inputtext = "Apostrophes: nation's Australia's other's another`s different´s apostrophes" \
                     "Hyphens: simple-hyphen ndash–hyphen mdash—hyphen horbar―hyphen ISBN 0-679-73392-2  1805–1872 (1856–1953) " \
                     "Infobox Football biography Defender/Sweeper Cabezón'' San Lorenzo de Almagro Chivas UAG Tecos" \
                     " Independiente Elche CF Club América San Lorenzo de Almagro 097 0(7) gold medal s in the nation's " \
                     "best in the nation's Australia's other's another`s different´s apostrophes rifled musket s," \
                     "rifled musket s, repeating rifles , and fortified entrenchment s contributed to the death of many" \
                     " men. General s and other officers , many professionally trained in tactics from the Napoleonic " \
                     "Wars , were often slow to develop changes in tactics in response. Outbreak of war companies " \
                     " ((each with roughly 100 men and led by a captain , with associated lieutenant s). Field" \
                     " officers normally included a colonel  (commanding), most frequent corps per army 1  4  2.74 2  divisions per (((.corps)  ( Rifle &amp; Light pages 180-81"

    for token in text2tokens(test_inputtext):
        print(token)

    # Test apostrophes
    print("---- apostophes test ----")
    apostrophes_test = "in the nation's Australia's other's another`s different´s apostrophes"
    print(remove_apostrophes(apostrophes_test))


if __name__ == "__main__":
    main()
