"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
import pickle
import re
import string
from collections import namedtuple
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import timedelta
from html.parser import HTMLParser
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Generator, Dict, Optional, Union, List

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
        self._previous_data, self._current_data = self._current_data, []

    def handle_endtag(self, tag):
        if tag == 'title':
            self._title = ''.join(self._current_data)
        elif tag == 'bdy':
            self._bdy = ''.join(self._current_data)
        elif tag == 'id' and self._previous_tag == 'title':
            self._title_id = ''.join(self._current_data)
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
        self.article_count: int = 0

    def populate(self, path: Union[str, PathLike[str]], articles_total: int = 281782):
        documents = Path(path).iterdir()
        # __benchmark__ {
        start = perf_counter()
        # __benchmark__ }
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(get_tokens_for_document, document): document for document in documents}
            for future in as_completed(futures):
                for article_title_id in future.result():
                    for token in future.result()[article_title_id]:
                        if token in self.tokens:
                            self.tokens[token].append(article_title_id)
                        else:
                            self.tokens[token] = [article_title_id]
                    self.article_count += 1
                    self.articles[article_title_id] = futures[future]
                    if self.article_count == articles_total:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                # __benchmark__ {
                articles_per_second = self.article_count / (perf_counter() - start)
                seconds_remaining = (articles_total - self.article_count) / articles_per_second
                print(f'\rIndexing: {self.article_count}/{articles_total} ({articles_per_second:.2f} articles/s)',
                      f'[Estimated remaining time: {timedelta(seconds=seconds_remaining)}]', end='')
                # __benchmark__ }
            print()
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
    with document.open(encoding='utf-8') as file:
        for line in file:
            parser.feed(line)
        yield from parser.articles
        parser.articles.clear()


def get_tokens_for_document(document: Path) -> dict[int, List[str]]:
    # TODO: move list comprehension to text2tokens
    return {article.title_id: [token for token in text2tokens(article.bdy)] for article in get_articles(document)}


def main():
    index = InvertedIndex()
    index.populate(Path('../dataset/wikipedia articles'))
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
