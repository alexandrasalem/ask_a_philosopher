import transformers
import torch

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
        return_full_text=True
    )
    print(outputs)
    text_output = outputs[0]["generated_text"][-1]
    return text_output
