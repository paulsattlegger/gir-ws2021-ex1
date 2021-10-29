"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
from time import perf_counter
from collections import namedtuple
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from typing import Generator

Article = namedtuple("Article", ["title", "title_id", "bdy"])


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title: str | None = None
        self.title_id: int | None = None
        self.bdy: str | None = None
        self.end_of_article: bool = False
        self._previous_tag, self._current_tag = None, None
        self._previous_data, self._current_data = None, None

    def handle_starttag(self, tag, attrs):
        self._previous_tag, self._current_tag = self._current_tag, tag
        self._previous_data, self._current_data = self._current_data, ''
        self.end_of_article = False

    def handle_endtag(self, tag):
        match tag:
            case 'title':
                self.title = self._current_data
            case 'bdy':
                self.bdy = self._current_data
            case 'id':
                match self._previous_tag:
                    case 'title':
                        self.title_id = int(self._current_data)
            case 'article':
                self.end_of_article = True

    def handle_data(self, data):
        match self._current_tag:
            case 'title' | 'bdy' | 'id':
                self._current_data += data


class InvertedIndex:
    def __init__(self):
        self.data: dict[str, list[int]] = {}

    def add(self, term: str, posting: int):
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
                    yield Article(parser.title, parser.title_id, parser.bdy)


def get_terms(article: Article) -> Generator[str, None, None]:
    for term in article.bdy.split(' '):
        yield term.strip()


def index_article(index: InvertedIndex, article: Article):
    for term in get_terms(article):
        index.add(term, article.title_id)


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
