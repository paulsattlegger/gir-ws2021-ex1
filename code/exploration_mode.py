"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""
from collections import Counter
from typing import Dict
import numpy as np
from scoring import TFIDFScoring, BM25Scoring
from createindex import InvertedIndex, text2tokens


def create_query_dict(query: str):
    tokens = text2tokens(query)
    return Counter(tokens)


def search(query_dict: Dict[str, int], index: InvertedIndex):
    results = []
    dfs = np.zeros(len(query_dict), dtype=int)
    for i, word in enumerate(query_dict):
        result = index.search(word)
        results.append(result)
        if result is None:
            dfs[i] = 0
        else:
            dfs[i] = len(result)
    return results, dfs


def main():
    index = InvertedIndex.load('../index.obj')
    article_count = index.article_count

    print("What do you want to search?")
    # query = input()
    query = "yellow yellow yellow dog england forest"
    query_dict = create_query_dict(query)
    results, dfs = search(query_dict, index)
    tfidf_scoring = TFIDFScoring(article_count)
    bm25_scoring = BM25Scoring(article_count)

    ranked = bm25_scoring.rank_articles(query_dict, results, dfs)
    article_title_ids = [ranked[i].key for i in range(min(len(ranked), 5))]
    articles = index.fetch(*article_title_ids)

    for article in articles:
        print("article id: ", article.title_id)
        print(article.bdy)


if __name__ == '__main__':
    main()
