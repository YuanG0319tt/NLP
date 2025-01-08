# -*- conding: utf-8 -*-
# @Time    : 2024/11/17  22:01
# @Author  : Yuan

import pandas as pd
import argparse
import pickle
from collections import defaultdict
import os

def parse_multi_word_argument(arg_list):
    return " ".join(arg_list)

def create_posting_list(phrases=[]):
    if os.path.exists("posting_list.pkl"):
        print("list already created.")
        return

    review_df = pd.read_pickle("reviews_segment.pkl")

    posting_list = defaultdict(list)

    for index, review in review_df.iterrows():
        review_text = review['review_text'].lower()

        for phrase in phrases:
            if phrase in review_text:
                posting_list[phrase].append(index)

        words = set(review_text.split())
        for word in words:
            posting_list[word].append(index)

    with open("posting_list.pkl", "wb") as file:
        pickle.dump(dict(posting_list), file)

    print("list created.")

def load_posting_list():

    with open("posting_list.pkl", "rb") as file:
        posting_list = pickle.load(file)
    return posting_list

def method1(aspect1, aspect2, opinion, posting_list):

    return set(posting_list.get(aspect1, [])) | set(posting_list.get(aspect2, [])) | set(posting_list.get(opinion, []))

def method2(aspect1, aspect2, opinion, posting_list):

    return set(posting_list.get(aspect1, [])) & set(posting_list.get(aspect2, [])) & set(posting_list.get(opinion, []))

def method3(aspect1, aspect2, opinion, posting_list):

    aspect_union = set(posting_list.get(aspect1, [])) | set(posting_list.get(aspect2, []))
    return aspect_union & set(posting_list.get(opinion, []))

def main():
    parser = argparse.ArgumentParser(description="Perform the boolean search.")

    parser.add_argument("-a1", "--aspect1", type=str, nargs='+', required=True, help="First word of the aspect (can be a phrase)")
    parser.add_argument("-a2", "--aspect2", type=str, nargs='+', required=True, help="Second word of the aspect (can be a phrase)")
    parser.add_argument("-o", "--opinion", type=str, nargs='+', required=True, help="Opinion word (can be a phrase)")
    parser.add_argument("-m", "--method", type=str, required=True, help="The method of boolean operation. Methods can be method1, method2 or method3")

    args = parser.parse_args()

    aspect1 = parse_multi_word_argument(args.aspect1)
    aspect2 = parse_multi_word_argument(args.aspect2)
    opinion = parse_multi_word_argument(args.opinion)

    create_posting_list(phrases=[aspect1, aspect2, opinion])

    posting_list = load_posting_list()

    if args.method.lower() == "method1":
        result = method1(aspect1, aspect2, opinion, posting_list)
    elif args.method.lower() == "method2":
        result = method2(aspect1, aspect2, opinion, posting_list)
    elif args.method.lower() == "method3":
        result = method3(aspect1, aspect2, opinion, posting_list)

    revs = pd.DataFrame()
    revs["review_index"] = [r for r in result]
    os.makedirs("output", exist_ok=True)
    output_filename = f"output/{aspect1}_{aspect2}_{opinion}_{args.method}.pkl"
    revs.to_pickle(output_filename)
    print(f"Output saved")

if __name__ == "__main__":
    main()
