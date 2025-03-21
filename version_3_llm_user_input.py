from llm import single_query_response
import streamlit as st


def main():
    question = st.text_input("Type your question for Aristotle below. An example is provided:", "How can I lead a virtuous life?")
    st.write("Asking Llama...")
    res = single_query_response(question)
    st.write(res)

if __name__ == "__main__":
    main()