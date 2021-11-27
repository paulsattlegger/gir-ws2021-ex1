"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""
from createindex import InvertedIndex
from collections import namedtuple
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Dict

from engine import Engine

# TODO: aim scoring TF-IDF 0.20 %, BM25 0.22-0.23 %
# TODO: parse topic file and get query id for relevant documents
# TODO: use trec_eval

Topic = namedtuple("Topic", ["query_id", "query_string"])


class TopicsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.topics: List[Topic] = []
        self._query_id: Optional[str] = None
        self._query_string: Optional[str] = None
        self._previous_tag, self._current_tag = None, None
        self._previous_data, self._current_data = None, None

    def handle_starttag(self, tag, attrs):
        self._previous_tag, self._current_tag = self._current_tag, tag
        self._previous_data, self._current_data = self._current_data, []
        if tag == 'topic':
            for name, value in attrs:
                if name == 'id':
                    self._query_id = value

    def handle_endtag(self, tag):
        if tag == 'title':
            self._query_string = ''.join(self._current_data)
        elif tag == 'topic':
            self.topics.append(Topic(self._query_id, self._query_string))

    def handle_data(self, data):
        if self._current_tag in ['title', 'bdy', 'id']:
            self._current_data.append(data)


def parse_topics_file(path: str):
    document = Path(path)
    parser = TopicsParser()
    file = document.open(encoding='utf-8')
    for line in file:
        parser.feed(line)
    return parser.topics


def compose_q_rel(results: Dict[str, Dict[int, float]], file: str):
    qrel = open(file, "w")
    name = "Test"
    for result in results.items():
        topic_id = result[0]
        postings = result[1]
        for i, posting_id in enumerate(postings):
            qrel.write(f"{topic_id} Q0 {posting_id} {i + 1} {postings[posting_id]} {name}\n")
    qrel.close()


def main():
    qrel_file = "../qrel.txt"
    topics = parse_topics_file('../dataset/topics.xml')
    print(topics)

    # TODO: change force_reindexing to True as specified in the assignment
    engine = Engine(force_reindexing=False)
    results = {}

    for topic in topics:
        print(f"searching for: {topic.query_string}")
        result = engine.search(topic.query_string)
        temp_res = {}
        for key in list(result)[0:100]:
            temp_res[key] = result[key]
        results[topic.query_id] = temp_res
    print("Saving Q-Rel file...")
    compose_q_rel(results, qrel_file)
    print("done.")
    # TODO: use trec_eval to evaluate


if __name__ == "__main__":
    main()
