import transformers
import torch
import pandas as pd

LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def single_query_response(question, prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would."):
    pipe = transformers.pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device="cuda:1",
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    outputs = pipe(
        messages,
        max_new_tokens = 512
    )
    text_output = outputs[0]["generated_text"][-1]
    return text_output

def multiple_query_responses(question_csv, prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would."):
    questions = pd.read_csv(question_csv, sep=",")
    queries = list(questions['Questions'])
    queries = [query.lower() for query in queries]

    responses = []
    for query in queries:
        text_output = single_query_response(query, prompt)['content']
        responses.append(text_output)

    questions['llm_response'] = responses
    return questions
