import transformers
import torch
import pandas as pd

LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def single_query_response(question):
    pipe = transformers.pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device="cuda:1",
    )
    messages = [
        {"role": "system", "content": "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would."},
        {"role": "user", "content": question},
    ]
    outputs = pipe(
        messages,
        max_new_tokens = 512
    )
    text_output = outputs[0]["generated_text"][-1]
    return text_output

def multiple_query_responses(question_csv):
    questions = pd.read_csv(question_csv, sep=",")
    queries = list(questions['Question'])
    queries = [query.lower() for query in queries]

    responses = []
    for query in queries:
        text_output = single_query_response(query)['content']
        responses.append(text_output)

    questions['llm_response'] = responses
    filename = f'{question_csv[:-4]}_responses.csv'
    questions.to_csv(filename, index=False)
    return responses
