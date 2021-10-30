"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""

from createindex import InvertedIndex

if __name__ == '__main__':
    index = InvertedIndex.load()

    queries = ['fine', 'art', 'england', '2000']
    postings = index.search(queries[0])
    article = index.fetch(postings[0])
    print(article.bdy)
