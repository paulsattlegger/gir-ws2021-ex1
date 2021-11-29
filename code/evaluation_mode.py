"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""
from collections import namedtuple
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from typing import Optional

from createindex import InvertedIndex
from engine import Engine

# TODO: aim scoring TF-IDF 0.20 %, BM25 0.22-0.23 %
# TODO: parse topic file and get query id for relevant documents
# TODO: use trec_eval

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


def compose_q_rel(results: dict[str, dict[int, float]], path: str):
    run_name = "Test"
    with Path(path).open("w") as file:
        for topic_id, postings in results.items():
            for i, posting_id in enumerate(postings):
                file.write(f"{topic_id} Q0 {posting_id} {i + 1} {postings[posting_id]} {run_name}\n")


def main():
    topics = parse_topics_file('../dataset/topics.xml')
    # TODO: change force_reindexing to True as specified in the assignment
    engine = Engine(force_reindexing=False)
    scoring_method = "tf-idf"
    results = {}

    for topic in topics:
        print(f"searching for: {topic.query_string}")
        result = engine.search(topic.query_string, scoring_method=scoring_method)
        temp_res = {}
        for key in islice(result, 100):
            temp_res[key] = result[key]
        results[topic.query_id] = temp_res
    print("Saving Q-Rel file...")
    compose_q_rel(results, f'../{scoring_method}.txt')
    print("done.")
    # TODO: use trec_eval to evaluate


if __name__ == "__main__":
    main()
