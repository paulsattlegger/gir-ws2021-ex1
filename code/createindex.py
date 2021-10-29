"""
This file contains your code to create the inverted index. Besides implementing and using the predefined tokenization function (text2tokens), there are no restrictions in how you organize this file.
"""
from collections import namedtuple
from nltk.stem import *
import re
import string

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
        word = remove_punctation(word)
        if word != "":  # remove "empty" words (if after removal nothing is left)
            yield from remove_hyphen(word)


def remove_apostrophes(token):
    # return token.split("'")[0]
    return re.split('\'|`|´', token)[0]


def remove_hyphen(token):
    """# remove hyphen only for words but not for numbers
    # (?<!...) is called negative lookbehind assertion https://docs.python.org/3/library/re.html
    for word in re.split('(?<!\d)[-|–|—|―](?!\d)', token):
        yield word"""
    for word in re.split('(?<=\d)[-|–|—|―](?<=\d)', token):
        yield word

def remove_punctation(token):
    #return re.split('\.|,|;', token)[0]
    # remove punctation only from words not for numbers
    # (?<!...) is called negative lookbehind assertion https://docs.python.org/3/library/re.html
    token = re.sub(r'(?<!\d)[^\w\s](?!\d)', '', token)
    return token.strip(string.punctuation) # to cover (1856–1953)


def stem(term):
    """
    This method uses the NLTK Porter stemmer
    :param term: A term string ready for stemming
    :return: the stemmed string
    """
    return PorterStemmer().stem(term)


def main():
    # TODO: aim creation time ~ 30 minutes
    index = InvertedIndex()
    index.add("Brutus", Posting("2", 1))
    index.add("Caesar", Posting("1", 1))
    index.add("Caesar", Posting("2", 1))
    print(index)

    """plurals = ['caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned', 'humbled', 'sized', 'meeting',
               'stating', 'siezing', 'itemization', 'sensational', 'traditional', 'reference', 'colonizer', 'plotted']

    singles = [stem(plural) for plural in plurals]
    print(' '.join(singles))"""

    testinputtext = "Apostrophes: nation's Australia's other's another`s different´s apostrophes" \
                    "Hyphens: simple-hyphen ndash–hyphen mdash—hyphen horbar―hyphen ISBN 0-679-73392-2  1805–1872 (1856–1953) " \
                    "Infobox Football biography Defender/Sweeper Cabezón'' San Lorenzo de Almagro Chivas UAG Tecos" \
                    " Independiente Elche CF Club América San Lorenzo de Almagro 097 0(7) gold medal s in the nation's " \
                    "best in the nation's Australia's other's another`s different´s apostrophes rifled musket s," \
                    "rifled musket s, repeating rifles , and fortified entrenchment s contributed to the death of many" \
                    " men. General s and other officers , many professionally trained in tactics from the Napoleonic " \
                    "Wars , were often slow to develop changes in tactics in response. Outbreak of war companies " \
                    " ((each with roughly 100 men and led by a captain , with associated lieutenant s). Field" \
                    " officers normally included a colonel  (commanding), most frequent corps per army 1  4  2.74 2  divisions per (((.corps)  ( Rifle &amp; Light pages 180-81"

    for token in text2tokens(testinputtext):
        print(token)

    # Test apostrophes
    print("---- apostophes test ----")
    apostrophesTest = "in the nation's Australia's other's another`s different´s apostrophes"
    print(remove_apostrophes(apostrophesTest))

    # TODO: parse HTML/XML files
    # TODO: save and load inverted index (pickle)


if __name__ == "__main__":
    main()
