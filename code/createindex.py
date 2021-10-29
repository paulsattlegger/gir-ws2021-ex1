"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
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


def main():
    articles = get_articles('../dataset/wikipedia articles')
    for article in islice(articles, 1):
        print(article)

    # TODO: aim creation time ~ 30 minutes
    index = InvertedIndex()
    index.add("Brutus", Posting("2", 1))
    index.add("Caesar", Posting("1", 1))
    index.add("Caesar", Posting("2", 1))

    # TODO: save and load inverted index (pickle)


if __name__ == "__main__":
    main()
