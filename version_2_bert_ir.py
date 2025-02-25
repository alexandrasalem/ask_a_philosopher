import argparse
from ir import ir_single_query_top_doc
import streamlit as st

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="type in a question you want to ask, surrounded by quotes")
    parser.add_argument("--use_streamlit", action="store_true", help="Whether to use streamlit")
    return parser.parse_args()

def main(question, use_streamlit=False):
    res = ir_single_query_top_doc(question, use_bert=True)
    if use_streamlit:
        st.write(res)
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))