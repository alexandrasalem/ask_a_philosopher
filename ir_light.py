import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embeddings_path):
    X = np.loadtxt(embeddings_path, delimiter=",")
    return X

def grab_bert_rep(sentences, tokenizer, model):
    rep_sentence = None
    count = 0
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state[:, -1, :] #outputs.pooler_output.detach().numpy().squeeze()
        last_hidden_state = torch.nn.functional.normalize(last_hidden_state, p=2, dim=1).squeeze()
        if rep_sentence is not None:
            rep_sentence = ((rep_sentence*count) + last_hidden_state)/(count+1)
            count += 1
        else:
            rep_sentence = last_hidden_state
            count+=1
    return rep_sentence

def calc_cos_sim(query_vector, X):
    res = cosine_similarity(X, query_vector).squeeze()
    return res

def ir_single_query_top_doc(question, embeddings_model, embeddings_tokenizer, corpus_embeddings, corpus_json):
    question = question.lower()
    with open(corpus_json, 'r') as file:
        data = json.load(file)

    last_hidden_state = grab_bert_rep([question], embeddings_tokenizer, embeddings_model)
    res = calc_cos_sim(last_hidden_state.detach().float().numpy().reshape(1, -1), corpus_embeddings)

    res_data = data[np.argmax(res)]
    sim = np.max(res)

    chapter_info = "I've discussed this topic in the following text: "

    if res_data['book_label'] == None:
        if res_data['chapter_label'] == None:
            chapter_info += f'{res_data["text_name"]}'
        else:
            chapter_info += f'{res_data["text_name"]}, {res_data["chapter_label"]}'
    else:
        if res_data['chapter_label'] == None:
            chapter_info += f'{res_data["text_name"]}, {res_data["book_label"]}'
        else:
            chapter_info += f'{res_data["text_name"]}, {res_data["book_label"]}, {res_data["chapter_label"]}'
    chapter_info = chapter_info + "\n\n"

    return res_data["chapter_text"], chapter_info, sim