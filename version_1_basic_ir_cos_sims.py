import argparse
from ir import ir_single_query_cos_sims

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="type in a question you want to ask, surrounded by quotes")
    return parser.parse_args()

def main(question):
    res = ir_single_query_cos_sims(question)
    return res

if __name__ == "__main__":
    args = get_args()
    main(**vars(args))