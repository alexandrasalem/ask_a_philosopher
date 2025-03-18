import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, BertModel
import torch
import re
import csv
from sklearn.metrics.pairwise import cosine_similarity

BERT_MODEL = "google-bert/bert-base-uncased" #"distilbert-base-uncased"
BERT_VECTORS = "bert_files.csv" #"distilbert_files.csv"

def load_tfidf(data):
    """
    This function creates a tfidf vectorizer and document-term matrix.

    Takes as input:
    :param data: corpus data, loaded as a dataframe, with 'chapter_text' column.
    And then outputs:
    :return: the vectorizer for the tfidf (vectorizer), and the document-term matrix (X)
    """

    corpus = [item['chapter_text'] for item in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()
    X = normalize(X, axis=1, norm="l1")
    return vectorizer, X

def load_bert(bert_reps_path = BERT_VECTORS):
    """
    This function creates a data variable tfidf vectorizer, and document-term matrix for a given json corpus.
    It expects a json file with 'chapter_text' in each element.

    Takes as input:
    :param corpus_json: location of json file with docs
    And then outputs:
    :return: the loaded corpus data (data), the vectorizer for the tfidf (vectorizer), and the document-term matrix (X)
    """

    X = np.loadtxt(bert_reps_path, delimiter=",") #pd.read_csv(bert_reps_path, header=None)
    return X


def grab_bert_rep(sentences, tokenizer, model):
    rep_sentence = None
    count = 0
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        last_hidden_state = outputs.pooler_output.detach().numpy().squeeze()
        if rep_sentence is not None:
            rep_sentence = ((rep_sentence*count) + last_hidden_state)/(count+1)
            count += 1
        else:
            rep_sentence = last_hidden_state
            count+=1
    return rep_sentence

def create_bert_corpus_reps(corpus_json):
    with open(corpus_json, 'r') as file:
        data = json.load(file)

    corpus = [item['chapter_text'] for item in data]
    ids = [item['id'] for item in data]

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = BertModel.from_pretrained(BERT_MODEL)

    chapter_lengths = []
    with open(BERT_VECTORS, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(corpus)):
            chapter_text = corpus[i]
            id = ids[i]
            chapter_split = re.split(r'[\!\?\.][ \n]', chapter_text)
            chapter_lengths.append(len(chapter_split))
            chapter_split = [item.lower() for item in chapter_split]
            last_hidden_states_avg = grab_bert_rep(chapter_split, tokenizer, model)
            writer.writerow([id]+last_hidden_states_avg.tolist())


def calc_cos_sim(query_vector, X):
    """
    This function calculates the cosine similarity between questions and answers.
    :param list_of_questions: a list of questions, each as a string
    :param tfidf_vectorizer: the vectorizer from tfidf
    :param X: The document-term matrix
    :return: a matrix of cosine similarities of size (# documents, # questions)
    """
    #X_norm = X/norm(X, axis=1)
    #query_vector_norm = query_vector/norm(query_vector, axis=1)
    #res = np.dot(X, query_vector).squeeze() / (norm(X, axis=1)*norm(query_vector))
    res = cosine_similarity(X, query_vector).squeeze()
    return res

def ir_single_query_cos_sims(question,  use_bert = False, corpus_json='aristotle.json'):
    """
    This function generates the cosine similarity values between the query and every doc in the corpus.
    Its output is in the form of a list of dictionaries (which can be converted to a json).

    Takes as input:
    :param question: the string of the current question
    :param corpus_json: location of json file with docs
    And then outputs:
    :return: the json with the element 'cos_sim_to_query' appended to each doc's item
    """
    question = question.lower()
    with open(corpus_json, 'r') as file:
        data = json.load(file)

    if use_bert:
        X = load_bert()
        X_edited = np.delete(X, 0, axis=1)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = BertModel.from_pretrained(BERT_MODEL, )
        inputs = tokenizer(question, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        last_hidden_state = outputs.pooler_output.detach().numpy()  # .squeeze()
        #last_hidden_state = normalize(last_hidden_state, axis=1, norm="l1")
        res = calc_cos_sim(last_hidden_state, X_edited)
    else:
        vectorizer, X = load_tfidf(data)
        query = [question]
        query_vector = vectorizer.transform(query)
        #query_vector = normalize(query_vector, axis=1, norm="l1")
        res = calc_cos_sim(query_vector.toarray(), X.toarray())
    res = list(np.squeeze(res))
    for i in range(len(data)):
        data[i]['cos_sim_to_query'] = res[i]
    return data

def ir_single_query_top_doc(question,  use_bert=False, corpus_json='aristotle.json'):
    """
    This function pulls the doc with the highest cosine similarity to the query and returns it.

    Takes as input:
    :param question: the string of the current question
    :param corpus_json: location of json file with docs
    :return: a string with the text of the doc with the highest cosine similarity to the query.
    """
    question = question.lower()
    with open(corpus_json, 'r') as file:
        data = json.load(file)

    if use_bert:
        X = load_bert()
        X_edited = np.delete(X, 0, axis=1)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = BertModel.from_pretrained(BERT_MODEL)
        inputs = tokenizer(question, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        last_hidden_state = outputs.pooler_output.detach().numpy()  # .squeeze()
        #last_hidden_state = normalize(last_hidden_state, axis=1, norm="l1")
        res = calc_cos_sim(last_hidden_state, X_edited)
    else:
        vectorizer, X = load_tfidf(data)
        query = [question]
        query_vector = vectorizer.transform(query)
        #query_vector = normalize(query_vector, axis=1, norm="l1")
        res = calc_cos_sim(query_vector.toarray(), X.toarray())

    res_data = data[np.argmax(res)]
    res_data_string = (f'You provided the following query: {question}\n\n'
                       f'Here is the closest chapter to that query:\n'
                       f'Text name: {res_data["text_name"]}\n\n'
                       f'Book name: {res_data["book_label"]}\n\n'
                       f'Chapter name: {res_data["chapter_label"]}\n\n'
                       f'Chapter text: {res_data["chapter_text"]}')
    return res_data_string

def ir_multiple_query_top_doc(question_csv,  use_bert=False, corpus_json='aristotle.json'):
    """
    This function pulls the doc with the highest cosine similarity to the query and returns it.

    Takes as input:
    :param question_csv: a csv with three columns: Question, Answer, Source_URL
    :param corpus_json: location of json file with docs
    :return: strings with the text of the docs with the highest cosine similarity to the queries. Printed as a series of text snippets.
    """
    with open(corpus_json, 'r') as file:
        data = json.load(file)

    questions = pd.read_csv(question_csv, sep=",")
    queries = list(questions['Question'])
    queries = [query.lower() for query in queries]

    if use_bert:
        X = load_bert()
        X_edited = np.delete(X, 0, axis=1)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = BertModel.from_pretrained(BERT_MODEL)
        inputs = tokenizer(queries, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        last_hidden_state = outputs.pooler_output.detach().numpy()  # .squeeze()
        #last_hidden_state = normalize(last_hidden_state, axis=1, norm="l1")
        res = calc_cos_sim(last_hidden_state, X_edited)
    else:
        vectorizer, X = load_tfidf(data)
        query_vectors = vectorizer.transform(queries)
        #query_vectors = normalize(query_vectors, axis=1, norm="l1")
        res = calc_cos_sim(query_vectors.toarray(), X.toarray())

    res_data = np.argmax(res, axis=0)
    res_data = res_data.astype(int)
    res_data = [data[i] for i in res_data]

    answers = list(questions['Answer'])
    ids = list(questions['Answer_id'])
    res_ids = [item["id"] for item in res_data]
    tot = len(res_ids)
    tot_corr = 0
    for j in range(len(res_ids)):
        if res_ids[j] == ids[j]:
            tot_corr += 1

    acc = tot_corr / tot
    res_data_string = f"You got {tot_corr} out of {tot} correct, giving {acc} accuracy.\n\n"
    res_data_string += "Below are answers to your queries:\n\n"
    for i in range(len(res_data)):
        res_data_string += (f'You provided the following query: {queries[i]}\n\n'
                            f'Here is the ground truth: {answers[i]}\n\n'
                            f'\n\n'
                            f'Here is the closest chapter to that query:\n\n'
                            f'Text name: {res_data[i]["text_name"]}\n\n'
                            f'Book name: {res_data[i]["book_label"]}\n\n'
                            f'Chapter name: {res_data[i]["chapter_label"]}\n\n'
                            f'Chapter text: {res_data[i]["chapter_text"]}\n\n'
                            f'-----------------------------------------\n\n')
    #print(res_data_string)
    return res_data_string

