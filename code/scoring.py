from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

ArticleWithScore = namedtuple("ArticleWithScore", "key score")


class Scoring(ABC):
    def __init__(self, total_num_articles, method = "cosine"):
        """
        Abstract class for scoring methods
        :param total_num_articles: total number of articles in database
        :param method: defines how the documents are compared to the query,
            "cosine": using cosine similarity
            "sum": just sums all scores for each term
        """
        assert method in ('cosine', 'sum'), f'Invalid method "{method}"'
        self._total_num_articles = total_num_articles
        self.method = method

    def _calc_query_vector(self, query: Dict[str, int], query_dict: dict, dfs: List[int]):
        dfs_np = np.array(dfs)
        query_vector = np.zeros(len(query_dict), dtype=float)
        idfs = np.where(dfs_np > 0, np.log(self._total_num_articles / dfs_np), 0)
        for i, word in enumerate(query):
            query_vector[i] = np.log(1 + query[word]) * idfs[i]
        return query_vector

    @abstractmethod
    def _calc_score(self, freq: NDArray[int], df: int):
        """
        Calculates scores for each document for one term
        :param freq: token frequency for each document found
        :param df: document frequency for the term
        """
        pass

    def _calc_score_for_all_articles(self, results):
        scored_docs: Dict[int, NDArray[float]] = {}
        for i, found_docs_by_word in enumerate(results):
            if found_docs_by_word is None or len(found_docs_by_word) == 0:
                continue

            token_freq = found_docs_by_word[:, 1]
            score = self._calc_score(token_freq, len(found_docs_by_word))

            for k, doc in enumerate(found_docs_by_word):
                if doc[0] in scored_docs:
                    scored_docs.get(doc[0])[i] = score[k]
                else:
                    scored_docs[doc[0]] = np.zeros(len(results), dtype=float)
                    scored_docs[doc[0]][i] = score[k]
        return scored_docs

    def _rank(self, articles: Dict[int, NDArray[float]], query_vector: NDArray[float]):
        ranked_articles: List[ArticleWithScore] = []
        norm_q_vec = np.linalg.norm(query_vector)

        for article_key in articles:
            if self.method == "cosine":
                norm_a_vec = np.linalg.norm(articles[article_key])
                score = np.dot(norm_a_vec, norm_q_vec)
            else:
                score = np.sum(articles[article_key])
            ranked_articles.append(ArticleWithScore(article_key, score))

        ranked_articles.sort(key=lambda a: a.score, reverse=True)
        return ranked_articles

    def rank_articles(self, query_dict: Dict[str, int], articles: List, dfs: List[int]) -> List[ArticleWithScore]:
        """
        Ranks all articles by importance
        :param query_dict: All occurred search terms and their frequency in the query
        :param articles: List of found articles by search term
        :param dfs: Document frequency for each search term
        :return: Ranked list of articles with their score
        """
        query_vector = self._calc_query_vector(query_dict, query_dict, dfs)
        articles_with_tf_idf = self._calc_score_for_all_articles(articles)
        ranked = self._rank(articles_with_tf_idf, query_vector)
        return ranked


class TFIDFScoring(Scoring):
    def _calc_score(self, freq: NDArray[int], df: int):
        tf = np.log(freq + 1)
        idf = np.log(self._total_num_articles / df)
        return tf * idf


class BM25Scoring(Scoring):
    def __init__(self,
                 total_num_articles: int,
                 method="cosine",
                 b: float = 0.75,
                 k: float = 1.25):
        """
        Okapi BM25 scoring method
        :param total_num_articles: total number of articles in database
        :param method: defines how the documents are compared to the query,
            "cosine": using cosine similarity
            "sum": just sums all scores for each term
        :param b: controls document length normalization
        :param k: controls term frequency scaling
        """
        super().__init__(total_num_articles, method)
        self.b = b
        self.k = k

    def _calc_score(self, freq: NDArray[int], df: int):
        adl = np.ones(len(freq))
        tf = (np.array(freq * (self.k + 1), dtype=float)
              / np.array(freq + self.k * (1 - self.b + self.b * adl), dtype=float))
        idf = np.log(self._total_num_articles / freq)
        return tf * idf
