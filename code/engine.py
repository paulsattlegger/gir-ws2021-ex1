from collections import Counter
from pathlib import Path
from typing import Dict

from createindex import InvertedIndex, text2tokens, Article
from scoring import TFIDFScoring, BM25Scoring


# TODO: this class isn't (yet ?) needed; remove if it doesn't extend index somehow
class Engine:
    def __init__(self,
                 index_file_path="../index.obj",
                 articles_path="../dataset/wikipedia articles",
                 force_reindexing=False):
        self._index_path = index_file_path
        index_file = Path(index_file_path)
        if not index_file.is_file() or force_reindexing:
            self._index = InvertedIndex()
            self._index.populate(articles_path)
            self._index.dump(index_file_path)
        else:
            self._index = InvertedIndex.load(index_file_path)
        self.article_count: int = self._index.article_count
        self.avg_article_len: float = self._index.avg_article_len

    @staticmethod
    def _create_query_dict(query: str):
        tokens = text2tokens(query)
        return Counter(tokens)

    def _retrieve_docs(self, query_dict: Dict[str, int]):
        results = [list(self._index.search(word)) for word in query_dict]
        dfs = list(map(lambda x: 0 if x is None else len(x), results))

        return results, dfs

    def search(self, query: str, scoring_method="bm25", ranking_method="sum"):
        assert scoring_method in ('tf-idf', 'bm25'), f'Invalid method "{scoring_method}"'
        query_dict = self._create_query_dict(query)
        results, dfs = self._retrieve_docs(query_dict)

        if scoring_method == "tf-idf":
            scoring = TFIDFScoring(self.article_count, self.avg_article_len, method=ranking_method)
        else:
            scoring = BM25Scoring(self.article_count, self.avg_article_len, method=ranking_method)

        ranked = scoring.rank_articles(query_dict, results, dfs)
        return ranked

    def fetch(self, article_title_id: int) -> Article:
        return self._index.fetch(article_title_id)
