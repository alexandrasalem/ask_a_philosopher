import transformers
import torch
import pandas as pd
from ir import ir_single_query_top_doc

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

def multiple_query_responses(question_csv, ir = True, prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would. Answer with 3-5 sentences at most."):
    questions = pd.read_csv(question_csv, sep=",")
    queries = list(questions['Questions'])
    queries = [query.lower() for query in queries]

    responses = []
    prompts = []
    for query in queries:
        if ir:
            doc = ir_single_query_top_doc(query)
            prompt = f"{prompt} Consider this relevant chapter from Aristotle's work when crafting your response: {doc}"
        text_output = single_query_response(query, prompt)['content']
        responses.append(text_output)
        prompts.append(prompt)

    questions['llm_response'] = responses
    questions['llm_prompt'] = prompts
    return questions
