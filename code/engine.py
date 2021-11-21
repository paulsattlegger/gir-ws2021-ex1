from abc import ABC
from collections import Counter
from typing import Dict
import numpy as np
from scoring import TFIDFScoring, BM25Scoring
from createindex import InvertedIndex, text2tokens
from pathlib import Path


class Engine(ABC):
    def __init__(self,
                 index_file_path="../index.obj",
                 articles_path="../dataset/wikipedia articles",
                 force_reindexing=False):
        self.index_path = index_file_path
        index_file = Path(index_file_path)
        if not index_file.is_file() or force_reindexing:
            self.index = InvertedIndex()
            self.index.populate(articles_path)
            self.index.dump(index_file_path)
        else:
            self.index = InvertedIndex.load(index_file_path)
        self.article_count = self.index.article_count

    @staticmethod
    def _create_query_dict(query: str):
        tokens = text2tokens(query)
        return Counter(tokens)

    def _retrieve_docs(self, query_dict: Dict[str, int]):
        results = []
        dfs = np.zeros(len(query_dict), dtype=int)
        for i, word in enumerate(query_dict):
            result = self.index.search(word)
            results.append(result)
            if result is None:
                dfs[i] = 0
            else:
                dfs[i] = len(result)
        return results, dfs

    def search(self, query: str, scoring_method="tf-idf"):
        assert scoring_method in ('tf-idf', 'bm25'), f'Invalid method "{scoring_method}"'
        query_dict = self._create_query_dict(query)
        results, dfs = self._retrieve_docs(query_dict)

        if scoring_method == "tf-idf":
            scoring = TFIDFScoring(self.article_count)
        else:
            scoring = BM25Scoring(self.article_count)

        ranked = scoring.rank_articles(query_dict, results, dfs)
        return ranked

    def search_and_print(self, query: str, scoring_method="tf-idf", num_results=3):
        ranked = self.search(query, scoring_method)
        article_title_ids = [ranked[i].key for i in range(min(len(ranked), num_results))]
        articles = self.index.fetch(*article_title_ids)

        for article in articles:
            print("article id: ", article.title_id)
            print(article.bdy)
