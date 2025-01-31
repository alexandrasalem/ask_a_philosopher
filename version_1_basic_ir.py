import argparse
from ir import ir_single_query_top_doc
import streamlit as st

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="type in a question you want to ask, surrounded by quotes")
    return parser.parse_args()

def main(question):
    res = ir_single_query_top_doc(question)
    st.write(res)
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))