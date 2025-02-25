from ir import ir_single_query_top_doc
import streamlit as st


def main():
    question = st.text_input("Type your question for Aristotle below. An example is provided:", "How can I lead a virtuous life?")
    res = ir_single_query_top_doc(question, use_bert=True)
    st.write(res)

if __name__ == "__main__":
    main()