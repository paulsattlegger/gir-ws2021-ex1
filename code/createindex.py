"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
from time import perf_counter
from collections import namedtuple, Counter
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from typing import Generator

Posting = namedtuple("Posting", ["document_id", "frequency"])
Article = namedtuple("Article", ["title", "bdy"])


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = None
        self.bdy = None
        self.end_of_article = False
        self._current_tag = None
        self._data = ''

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag
        self._data = ''
        self.end_of_article = False

    def handle_endtag(self, tag):
        match tag:
            case 'title':
                self.title = self._data
            case 'bdy':
                self.bdy = self._data
            case 'article':
                self.end_of_article = True

    def handle_data(self, data):
        match self._current_tag:
            case 'title' | 'bdy':
                self._data += data


class InvertedIndex:
    def __init__(self):
        self.data: dict[str, list[Posting]] = {}

    def add(self, term: str, posting: Posting):
        if term in self.data:
            self.data[term].append(posting)
        else:
            self.data[term] = [posting]

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
    pass


def get_articles(path: str) -> Generator[Article, None, None]:
    parser = ArticleParser()
    path = Path(path)
    for file in path.iterdir():
        with open(file, encoding='utf-8') as fh:
            for line in fh:
                parser.feed(line)
                if parser.end_of_article:
                    yield Article(parser.title, parser.bdy)


def get_terms(article: Article) -> Generator[str, None, None]:
    for term in article.bdy.split(' '):
        yield term.strip()


def index_article(index: InvertedIndex, article: Article):
    words = get_terms(article)
    for term, count in Counter(words).items():
        index.add(term, Posting(article.title, count))


def main():
    index = InvertedIndex()
    articles = get_articles('../dataset/wikipedia articles')

    # Benchmark
    start = perf_counter()
    n = 10000
    for article in islice(articles, n):
        index_article(index, article)
    articles_total = 283438
    articles_per_second = n / (perf_counter() - start)
    print(f'{articles_per_second} articles/s')
    print(f'{articles_total / articles_per_second / 60} m total')

    # TODO: aim creation time ~ 30 minutes
    # TODO: save and load inverted index (pickle)


if __name__ == "__main__":
    main()
