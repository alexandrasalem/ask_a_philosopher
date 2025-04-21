import argparse
from llm import multiple_query_responses
import streamlit as st
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question_csv", type=str, help="Location of csv file with questions")
    parser.add_argument("--use_streamlit", action="store_true", help="Whether to use streamlit")
    return parser.parse_args()

def main(question_csv, use_streamlit=False):
    res = multiple_query_responses(question_csv)
    if use_streamlit:
        st.write(res)
    print(res)
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))