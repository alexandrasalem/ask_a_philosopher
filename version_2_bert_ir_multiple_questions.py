import argparse
from ir import ir_multiple_query_top_doc
import streamlit as st

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question_csv", type=str, help="Location of csv file with questions")
    parser.add_argument("--use_streamlit", action="store_true", help="Whether to use streamlit")
    return parser.parse_args()

def main(question_csv, use_streamlit=False):
    res = ir_multiple_query_top_doc(question_csv, use_bert=True)
    if use_streamlit:
        st.write(res)
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))