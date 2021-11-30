from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from createindex import Posting


class Scoring:
    def __init__(self, total_num_articles: int, avg_doc_length: float, method: str = "cosine"):
        """
        Abstract class for scoring methods
        :param total_num_articles: total number of articles in database
        :param avg_doc_length: average length of all documents in database
        :param method: defines how the documents are compared to the query,
            "cosine": using cosine similarity
            "sum": just sums all scores for each term
        """
        assert method in ('cosine', 'sum'), f'Invalid method "{method}"'
        self._total_num_articles = total_num_articles
        self._avg_doc_length = avg_doc_length
        self._method = method

    def _calc_query_vector(self, query: dict[str, int], query_dict: dict, dfs: list[int]):
        dfs_np = np.array(dfs)
        query_vector = np.zeros(len(query_dict), dtype=float)
        idfs = np.where(dfs_np > 0, np.log(self._total_num_articles / dfs_np), 0)
        for i, word in enumerate(query):
            query_vector[i] = np.log(1 + query[word]) * idfs[i]
        return query_vector

    @abstractmethod
    def _calc_score(self, freq: NDArray[int], df: int, doc_length: NDArray[float]):
        """
        Calculates scores for each document for one term
        :param freq: token frequency for each document found
        :param df: document frequency for the term
        :param doc_length: document length / average document length for each document
        """
        pass

    def _calc_score_for_all_articles(self, results: list[list[Posting]]):
        scored_docs: dict[int, NDArray[float]] = {}
        for i, found_docs_by_word in enumerate(results):
            if found_docs_by_word is None or len(found_docs_by_word) == 0:
                continue

            token_freq: NDArray[int] = np.array(list(map(lambda p: p.tf, found_docs_by_word)))
            doc_length = np.array(list(map(lambda p: p.article_len, found_docs_by_word)), dtype=float)
            doc_length = doc_length / self._avg_doc_length
            score = self._calc_score(token_freq, len(found_docs_by_word), doc_length)

            for k, doc in enumerate(found_docs_by_word):
                if doc[0] in scored_docs:
                    scored_docs.get(doc[0])[i] = score[k]
                else:
                    scored_docs[doc[0]] = np.zeros(len(results), dtype=float)
                    scored_docs[doc[0]][i] = score[k]
        return scored_docs

    def _rank(self, articles: dict[int, NDArray[float]], query_vector: NDArray[float]):
        scores: dict[int, float] = {}
        norm_q_vec = np.linalg.norm(query_vector)

        for article_title_id in articles:
            if self._method == "cosine":
                norm_a_vec = np.linalg.norm(articles[article_title_id])
                score = np.dot(norm_a_vec, norm_q_vec)
            else:
                score = np.sum(articles[article_title_id])
            scores[article_title_id] = float(score)

        # https://stackoverflow.com/a/61793402
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def rank_articles(self,
                      query_dict: dict[str, int],
                      articles: list[list[Posting]],
                      dfs: list[int]) -> dict[int, float]:
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
    def _calc_score(self, freq: NDArray[int], df: int, doc_length: NDArray[float]):
        tf = np.log(freq + 1)
        idf = np.log(self._total_num_articles / df)
        return tf * idf


class BM25Scoring(Scoring):
    def __init__(self,
                 total_num_articles: int,
                 avg_doc_length: float,
                 method="sum",
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
        super().__init__(total_num_articles, avg_doc_length, method)
        self.b = b
        self.k = k

    def _calc_score(self, freq: NDArray[int], df: int, doc_length: NDArray[float]):
        """
        Formula from: https://en.wikipedia.org/wiki/Okapi_BM25
        """
        numerator = np.array(freq * (self.k + 1), dtype=float)
        denominator = np.array(freq + self.k * (1 - self.b + self.b * doc_length), dtype=float)
        tf = numerator / denominator
        idf = np.log((self._total_num_articles - df + 0.5)/(df + 0.5) + 1)
        return tf * idf
