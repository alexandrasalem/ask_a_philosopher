import argparse
from ir import ir_single_query_cos_sims
import pandas as pd
import streamlit as st

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="type in a question you want to ask, surrounded by quotes")
    parser.add_argument("--use_streamlit", action="store_true", help="Whether to use streamlit")
    return parser.parse_args()

def main(question, use_streamlit=False):
    res = ir_single_query_cos_sims(question)
    df = pd.DataFrame(res)
    df['text_book_chap'] = df["text_name"] + df["book_label"].astype(str) + df["chapter_label"]
    if use_streamlit:
        st.write(f'Here is your question: {question}')
        st.write(df)
        st.line_chart(df, x='text_book_chap', y='cos_sim_to_query')
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))