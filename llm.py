import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
#from ir import ir_single_query_top_doc


def single_query_response(question, model, tokenizer, prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would."):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=512,
            do_sample=True,
        )

    text_output = tokenizer.decode(
    outputs[0][input_ids["input_ids"].shape[-1]:],
    skip_special_tokens=True
)
    # pipe = transformers.pipeline(
    #     "text-generation",
    #     model=LLM_MODEL_ID,
    #     torch_dtype=torch.bfloat16,
    #     device=device,
    # )
    # messages = [
    #     {"role": "system", "content": prompt},
    #     {"role": "user", "content": question},
    # ]
    # outputs = pipe(
    #     messages,
    #     max_new_tokens = 512
    # )
    # text_output = outputs[0]["generated_text"][-1]
    return text_output

# def multiple_query_responses(question_csv, output_filename, ir = True, prompt = "You are the ancient philosopher, Aristotle. Respond to this question as Aristotle would. Answer with 3-5 sentences at most."):
#     questions = pd.read_csv(question_csv, sep=",")
#     queries = list(questions['Questions'])
#     queries = [query.lower() for query in queries]
#
#     responses = []
#     prompts = []
#     for query in queries:
#         if ir:
#             doc = ir_single_query_top_doc(query)
#             prompt = f"{prompt} Consider this relevant chapter from Aristotle's work when crafting your response: {doc}"
#         text_output = single_query_response(query, prompt)['content']
#         responses.append(text_output)
#         prompts.append(prompt)
#     # with open(output_filename, "a") as f:
#     #     i = 0
#     #     for query in queries:
#     #         if ir:
#     #             doc = ir_single_query_top_doc(query)
#     #             prompt = f"{prompt} Consider this relevant chapter from Aristotle's work when crafting your response: {doc}"
#     #         text_output = single_query_response(query, prompt)['content']
#     #         f.write(f'{i}, {text_output}\n')
#     #         f.flush()
#     #         i+=1
#     #         responses.append(text_output)
#     #         prompts.append(prompt)
#
#
#     questions['llm_response'] = responses
#     questions['llm_prompt'] = prompts
#     return questions

# print(single_query_response("What is the meaning of life?"))