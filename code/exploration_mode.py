"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""
from abc import ABC
from collections import Counter, namedtuple
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from createindex import InvertedIndex, text2tokens
ArticleWithScore = namedtuple("ArticleWithScore", "key score")

class Scoring(ABC):
    def __init__(self, total_num_articles):
        self._total_num_articles = total_num_articles

    def _calc_query_vector(self, query: Dict[str, int], query_dict: dict, dfs: NDArray[int]):
        query_vector = np.zeros(len(query_dict), dtype=float)
        idfs = np.where(dfs > 0, np.log(self._total_num_articles / dfs), 0)
        for i, word in enumerate(query):
            query_vector[i] = np.log(1 + query[word]) * idfs[i]
        return query_vector

    @staticmethod
    def _calc_tf_idf(token_count: NDArray[int], total_num_articles: int, df: int):
        tf = np.log(token_count + 1)
        idf = np.log(total_num_articles / df)
        return tf * idf

    def _calc_tf_idf_for_all_articles(self, results):
        articles_with_tf_idf: Dict[int, NDArray[float]] = {}
        for i, found_articles_by_word in enumerate(results):
            if found_articles_by_word is None or len(found_articles_by_word) == 0:
                continue
            tf_idf = self._calc_tf_idf(found_articles_by_word[:, 1],
                                       self._total_num_articles,
                                       len(found_articles_by_word))
            # results: [[article_id, token_occurrences, tf_idf]]
            # word_result = np.insert(word_result, 2, tf_idf, axis=1)

            for k, article in enumerate(found_articles_by_word):
                if article[0] in articles_with_tf_idf:
                    articles_with_tf_idf.get(article[0])[i] = tf_idf[k]
                else:
                    articles_with_tf_idf[article[0]] = np.zeros(len(results), dtype=float)
                    articles_with_tf_idf[article[0]][i] = tf_idf[k]
        return articles_with_tf_idf

    @staticmethod
    def _rank(articles: Dict[int, NDArray[float]], query_vector: NDArray[float], method="cosine"):
        assert method in ('cosine', 'sum'), f'Invalid method "{method}"'
        ranked_articles: List[ArticleWithScore] = []
        norm_q_vec = np.linalg.norm(query_vector)

        for article_key in articles:
            if method == "cosine":
                norm_a_vec = np.linalg.norm(articles[article_key])
                score = np.dot(norm_a_vec, norm_q_vec)
            else:
                score = np.sum(articles[article_key])
            ranked_articles.append(ArticleWithScore(article_key, score))

        ranked_articles.sort(key=lambda a: a.score, reverse=True)
        return ranked_articles

    def rank_articles(self, query_dict: Dict[str, int], articles, dfs) -> List[ArticleWithScore]:
        query_vector = self._calc_query_vector(query_dict, query_dict, dfs)
        articles_with_tf_idf = self._calc_tf_idf_for_all_articles(articles)
        ranked = self._rank(articles_with_tf_idf, query_vector)
        return ranked


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
    scoring = Scoring(article_count)
    ranked = scoring.rank_articles(query_dict, results, dfs)
    article_title_ids = [ranked[i].key for i in range(min(len(ranked), 5))]
    articles = index.fetch(*article_title_ids)

    for article in articles:
        print("article id: ", article.title_id)
        print(article.bdy)


if __name__ == '__main__':
    main()
