from llm import single_query_response
from ir import ir_single_query_top_doc
import runpod
from huggingface_hub import login
import os
import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#hf_token = os.environ['hf_token']
#login(token=hf_token)

LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID
)
model.to(device)

model.eval()

print("✅ Model loaded and ready.")


#prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would. Keep your response very short."

def process_input_llm(question, prompt):
    prompt = prompt + " Keep your response very short."
    llm_res = single_query_response(question, model = model, tokenizer = tokenizer, prompt = prompt)
    res = llm_res#['content']
    return res

def process_input_rag(question, prompt, corpus):
    doc, info, sim = ir_single_query_top_doc(question, use_bert=False, corpus_json=corpus)
    prompt = f"{prompt} Consider this relevant chapter from Aristotle's work when crafting your response: {doc}"
    prompt = prompt + "\n\nKeep your response very short."
    llm_res = single_query_response(question, model = model, tokenizer = tokenizer, prompt = prompt)
    res = info
    res += llm_res
    return res, sim


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
    mode = input.get('mode')
    prompt = f"You are the ancient philosopher, {philosopher}. Respond to this question as {philosopher} would."
    #seconds = input.get('seconds', 0)

    print(f"Received question: {question}")
    #print(f"Sleeping for {seconds} seconds...")

    if mode == "LLM-only":
        res = process_input_llm(question = question, prompt = prompt)
        sim = None
    else:
        if philosopher == "Aristotle":
            res, sim = process_input_rag(question=question, prompt=prompt, corpus="aristotle.json")
        else:
            res, sim = process_input_rag(question=question, prompt=prompt, corpus="confucius.json")
    return res, sim


# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})