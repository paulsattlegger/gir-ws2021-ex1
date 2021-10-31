"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""

from createindex import InvertedIndex
import numpy as np

if __name__ == '__main__':
    index = InvertedIndex.load()

    queries = ['fine', 'art', 'england', '2000']

    ar1 = index.search(queries[0])
    for i in range(1, len(queries)):
        ar2 = index.search(queries[i])
        ar1 = np.intersect1d(ar1, ar2)

    article_1 = int(ar1[0])
    print(index.fetch(article_1).bdy)
