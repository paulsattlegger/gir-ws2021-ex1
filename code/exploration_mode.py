"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked
list of documents is returned. Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
from shutil import get_terminal_size
from time import perf_counter
from typing import Dict

from createindex import text2tokens, InvertedIndex
from scoring import Scoring, BM25Scoring, TFIDFScoring


def _create_query_dict(query: str):
    tokens = text2tokens(query)
    return Counter(tokens)


def _retrieve_docs(index: InvertedIndex, query_dict: Dict[str, int]):
    results = [list(index.search(word)) for word in query_dict]
    dfs = list(map(lambda x: 0 if x is None else len(x), results))

    return results, dfs


def search(index: InvertedIndex, query: str, scoring: Scoring):
    query_dict = _create_query_dict(query)
    results, dfs = _retrieve_docs(index, query_dict)

    ranked = scoring.rank_articles(query_dict, results, dfs)
    return ranked


def main():
    index = InvertedIndex.load('../index.obj')
    scoring = BM25Scoring(index.article_count, index.avg_article_len)

    while True:
        print("Commands ':q' = quit | :t' = tf-idf-scoring | ':b' = bm25-scoring")
        print(f"Current scoring method: {scoring}")
        print("What do you want to search?")
        query = input('> ')

        if query == ":q":
            break
        elif query == ":t":
            scoring = TFIDFScoring(index.article_count, index.avg_article_len)
        elif query == ":b":
            scoring = BM25Scoring(index.article_count, index.avg_article_len)
        else:
            start = perf_counter()
            result = search(index, query, scoring)
            print(f'{len(result)} results ({timedelta(seconds=perf_counter() - start).microseconds / 1000} ms)')
            for i, (article_title_id, score) in enumerate(result.items()):
                article = index.fetch(article_title_id)
                print(f"Result #: {i + 1}")
                print(f"Title: {article.title} | ID: {article.title_id} | Score: {score}")
                print(f"{'-' * get_terminal_size().columns}")
                print(f"{article.bdy}")
                print(f"{'-' * get_terminal_size().columns}")
                print(f"Commands ':n' = next result | '<any>' = new search ")
                query = input('> ')
                if query != ":n":
                    break

    print("bye.")


if __name__ == '__main__':
    main()
