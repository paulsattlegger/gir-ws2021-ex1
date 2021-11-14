"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""

from createindex import InvertedIndex
import numpy as np

if __name__ == '__main__':
    index = InvertedIndex.load('../index.obj')
    article_count = index.article_count

    print("What do you want to search?")
    query = input()
    found = index.search(query)
    if found is not None and len(found) > 0:
        sortedArr = found[found[:, 1].argsort()]
        sortedArr = sortedArr[::-1]

        for i in range(min(len(sortedArr), 5)):
            article = index.fetch(sortedArr[i][0])
            print("article id: ", sortedArr[i][0])
            print(article.bdy)
    else:
        print("Nothing found")

