"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
from collections import namedtuple

# TODO: from html.parser import HTMLParser

Posting = namedtuple("Posting", ["document_id", "frequency"])


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


def main():
    # TODO: aim creation time ~ 30 minutes
    index = InvertedIndex()
    index.add("Brutus", Posting("2", 1))
    index.add("Caesar", Posting("1", 1))
    index.add("Caesar", Posting("2", 1))
    print(index)

    # TODO: parse HTML/XML files
    # TODO: save and load inverted index (pickle)


if __name__ == "__main__":
    main()
