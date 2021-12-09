"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import namedtuple
from datetime import timedelta
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Optional

from createindex import InvertedIndex
from exploration_mode import search
from scoring import BM25Scoring, TFIDFScoring

Topic = namedtuple("Topic", ["query_id", "query_string"])


class TopicsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.topics: list[Topic] = []
        self._query_id: Optional[str] = None
        self._query_string: Optional[str] = None
        self._tag = None
        self._data = []

    def handle_starttag(self, tag, attrs):
        self._tag = tag
        self._data = []
        if tag == 'topic':
            for name, value in attrs:
                if name == 'id':
                    self._query_id = value

    def handle_endtag(self, tag):
        if tag == 'title':
            self._query_string = ''.join(self._data)
        elif tag == 'topic':
            self.topics.append(Topic(self._query_id, self._query_string))

    def handle_data(self, data):
        if self._tag == 'title':
            self._data.append(data)


def parse_topics_file(path: str):
    document = Path(path)
    parser = TopicsParser()
    file = document.open(encoding='utf-8')
    for line in file:
        parser.feed(line)
    return parser.topics


def compose_q_rel(results: dict[str, dict[int, float]], scoring):
    with Path(f'../retrieval_results/{scoring}_title_only.txt').open("w") as file:
        for topic_id, postings in results.items():
            for i, posting_id in enumerate(postings):
                file.write(f"{topic_id} Q0 {posting_id} {i + 1} {postings[posting_id]} {scoring}\n")


def main():
    index = InvertedIndex.load('../index.obj')
    topics = parse_topics_file('../dataset/topics.xml')

    method = "cosine" if args.cosine else "sum"
    if args.bm25:
        scoring = BM25Scoring(index.article_count, index.avg_article_len, method)
    else:
        scoring = TFIDFScoring(index.article_count, index.avg_article_len, method)
    results = {}

    n = 100
    times = []
    for topic in topics:
        print(f"searching for: {topic.query_string}")
        start = perf_counter()
        result = search(index, topic.query_string, scoring)
        times.append(perf_counter() - start)
        temp_res = {}
        for key in islice(result, n):
            temp_res[key] = result[key]
        results[topic.query_id] = temp_res
    print(f'Average query time: {timedelta(seconds=sum(times) / n)}')
    print("Saving Q-Rel file...")
    compose_q_rel(results, scoring)
    print("done.")


def parse_args():
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', '--bm25', action='store_true', help='use BM25 scoring')
    group.add_argument('-t', '--tf-idf', action='store_true', help='use TF-IDF scoring')
    parser.add_argument('-c', '--cosine', action='store_true',
                        help='use cosine similarity to compare documents to queries, '
                             'normally the sum of all scores is used')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main()
