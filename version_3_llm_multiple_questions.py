import argparse
import os
from llm import multiple_query_responses
import streamlit as st
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question_csv", type=str, help="Location of csv file with questions")
    parser.add_argument("--use_streamlit", action="store_true", help="Whether to use streamlit")
    return parser.parse_args()

def main(question_csv, use_streamlit=False):
    res = multiple_query_responses(question_csv)
    current_date = datetime.now()
    date_string = current_date.strftime("%Y-%m-%d")
    os.makedirs(date_string, exist_ok=False)
    filename = f'{date_string}/{question_csv[:-4]}_responses.csv'
    res.to_csv(filename, index=False)
    if use_streamlit:
        st.write(res)
    print(res)
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))