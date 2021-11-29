"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked
list of documents is returned. Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""
from datetime import timedelta
from shutil import get_terminal_size
from time import perf_counter

from createindex import InvertedIndex
from engine import Engine


def main():
    engine = Engine()
    method = "bm25"

    while True:
        print("Commands ':q' = quit | :t' = tf-idf-scoring | ':b' = bm25-scoring")
        print(f"Current scoring method: {method}")
        print("What do you want to search?")
        query = input('> ')

        if query == ":q":
            break
        elif query == ":t":
            method = "tf-idf"
        elif query == ":b":
            method = "bm25"
        else:
            start = perf_counter()
            result = engine.search(query, method)
            print(f'{len(result)} results ({timedelta(seconds=perf_counter() - start).microseconds / 1000} ms)')
            for i, (article_title_id, score) in enumerate(result.items()):
                article = engine.fetch(article_title_id)
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
