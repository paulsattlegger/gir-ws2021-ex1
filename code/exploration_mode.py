"""
This file contains your code for the interactive exploration mode where a string can be input by a user and a ranked list of documents is returned.
Make sure that the user can switch between TF-IDF and BM25 scoring functions.
"""

from engine import Engine


def main():
    engine = Engine()
    method = "tf-idf"

    while True:
        print("Commands ':q' = quit | ':t' = tf-idf-scoring | ':b' = bm25-scoring")
        print(f"Current scoring method: {method}")
        print("What do you want to search?")
        query = input()
        if query == ":q":
            break
        elif query == ":t":
            method = "tf-idf"
        elif query == ":b":
            method = "bm25"
        else:
            engine.search_and_print(query, scoring_method=method)

    print("bye.")


if __name__ == '__main__':
    main()
