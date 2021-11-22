from abc import ABC
from collections import Counter
from createindex import InvertedIndex, text2tokens
from typing import Dict
from scoring import TFIDFScoring, BM25Scoring
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
        self.article_count: int = self.index.article_count
        self.avg_article_len: float = self.index.avg_article_len

    @staticmethod
    def _create_query_dict(query: str):
        tokens = text2tokens(query)
        return Counter(tokens)

    def _retrieve_docs(self, query_dict: Dict[str, int]):
        results = [list(self.index.search(word)) for word in query_dict]
        dfs = list(map(lambda x: 0 if x is None else len(x), results))

        return results, dfs

    def search(self, query: str, scoring_method="bm25"):
        assert scoring_method in ('tf-idf', 'bm25'), f'Invalid method "{scoring_method}"'
        query_dict = self._create_query_dict(query)
        results, dfs = self._retrieve_docs(query_dict)

        if scoring_method == "tf-idf":
            scoring = TFIDFScoring(self.article_count, self.avg_article_len)
        else:
            scoring = BM25Scoring(self.article_count, self.avg_article_len)

        ranked = scoring.rank_articles(query_dict, results, dfs)
        return ranked

    def search_and_print(self, query: str, scoring_method="bm25", num_results=3):
        ranked = self.search(query, scoring_method)
        article_keys = [int(ranked[i].key) for i in range(min(len(ranked), num_results))]
        articles = list(self.index.fetch(*article_keys))

        for i, article in enumerate(articles):
            print(f"Result number: {i + 1}")
            print(f"Article id: {article.title_id} | Score: {ranked[i].score}")
            print("-"*100)
            print(article.bdy)
            print("-"*100)
            print("\n\n")
