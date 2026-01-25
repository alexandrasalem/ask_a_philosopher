from llm import single_query_response
from ir import ir_single_query_top_doc
import runpod
from huggingface_hub import login
import os
import streamlit as st

hf_token = os.environ['HF_TOKEN']
login(token=hf_token)

#prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would. Keep your response very short."

def process_input_llm(question, prompt):
    llm_res = single_query_response(question, prompt = prompt)
    res = llm_res['content']
    return res

def process_input_rag(question, prompt):
    doc = ir_single_query_top_doc(question)
    prompt = f"{prompt} Consider this relevant chapter from Aristotle's work when crafting your response: {doc}"
    llm_res = single_query_response(question, prompt = prompt)
    #res = f"Your question: {question}\n"
    #res += "Answer:\n"
    #res += llm_res['content']
    res = llm_res['content']
    return res


def handler(event):
    #   This function processes incoming requests to your Serverless endpoint.
    #
    #    Args:
    #        event (dict): Contains the input data and request metadata
    #
    #    Returns:
    #       Any: The result to be returned to the client

    # Extract input data
    print(f"Worker Start")
    input = event['input']

    question = input.get('question')
    philosopher = input.get('philosopher')
    rag = input.get('rag')
    prompt = f"You are the ancient philosopher, {philosopher}. Respond to this question as {philosopher} would. Keep your response very short."
    #seconds = input.get('seconds', 0)

    print(f"Received question: {question}")
    #print(f"Sleeping for {seconds} seconds...")

    # You can replace this sleep call with your own Python code
    if rag == "No":
        res = process_input_llm(question = question, prompt = prompt)
    else:
        res = process_input_rag(question=question, prompt=prompt)
    return res


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})