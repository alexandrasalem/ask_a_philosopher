import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

def load_tfidf(corpus_json):
    """
    This function creates a data variable tfidf vectorizer, and tfidf matrix for a given json corpus.
    It expects a json file with 'chapter_text' in each element.

    Takes as input:
    :param corpus_json: location of json file with docs
    And then outputs:
    :return: the loaded corpus data (data), the vectorizer for the tfidf (vectorizer), and the tfidf matrix (X)
    """
    with open(corpus_json, 'r') as file:
        data = json.load(file)

    corpus = [item['chapter_text'] for item in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()
    X = normalize(X, axis=1, norm="l1")
    return data, vectorizer, X

def ir_single_query_cos_sims(question,  corpus_json='aristotle.json'):
    """
    This function generates the cosine similarity values between the query and every doc in the corpus.
    Its output is in the form of a list of dictionaries (which can be converted to a json).

    Takes as input:
    :param question: the string of the current question
    :param corpus_json: location of json file with docs
    And then outputs:
    :return: the json with the element 'cos_sim_to_query' appended to each doc's item
    """
    data, vectorizer, X = load_tfidf(corpus_json)
    query = [question]
    query_vector = vectorizer.transform(query)
    query_vector = normalize(query_vector, axis=1, norm="l1")

    # Now the multiplication
    res = np.matmul(X.toarray(), np.transpose(query_vector.toarray()))
    res = list(np.squeeze(res))
    for i in range(len(data)):
        data[i]['cos_sim_to_query'] = res[i]
    return data

def ir_single_query_top_doc(question,  corpus_json='aristotle.json'):
    """
    This function pulls the doc with the highest cosine similarity to the query and returns it.

    Takes as input:
    :param question: the string of the current question
    :param corpus_json: location of json file with docs
    :return: a string with the text of the doc with the highest cosine similarity to the query.
    """
    data, vectorizer, X = load_tfidf(corpus_json)
    query = [question]
    query_vector = vectorizer.transform(query)
    query_vector = normalize(query_vector, axis=1, norm="l1")

    # Now the multiplication
    res = np.matmul(X.toarray(), np.transpose(query_vector.toarray()))

    res_data = data[np.argmax(res)]
    res_data_string = (f'You provided the following query: {question}\n\n'
                       f'Here is the closest chapter to that query:\n'
                       f'Text name: {res_data["text_name"]}\n\n'
                       f'Book name: {res_data["book_label"]}\n\n'
                       f'Chapter name: {res_data["chapter_label"]}\n\n'
                       f'Chapter text: {res_data["chapter_text"]}')
    return res_data_string

def ir_multiple_query_top_doc(question_csv,  corpus_json='aristotle.json'):
    """
    This function pulls the doc with the highest cosine similarity to the query and returns it.

    Takes as input:
    :param question_csv: a csv with three columns: Question, Answer, Source_URL
    :param corpus_json: location of json file with docs
    :return: strings with the text of the docs with the highest cosine similarity to the queries. Printed as a series of text snippets.
    """
    data, vectorizer, X = load_tfidf(corpus_json)
    questions = pd.read_csv(question_csv, sep=",")

    queries = list(questions['Question'])
    answers = list(questions['Answer'])
    ids = list(questions['Answer_id'])
    query_vector = vectorizer.transform(queries)
    query_vector = normalize(query_vector, axis=1, norm="l1")

    # Now the multiplication
    res = np.matmul(X.toarray(), np.transpose(query_vector.toarray()))
    res_data = np.argmax(res, axis=0)
    res_data = res_data.astype(int)
    res_data = [data[i] for i in res_data]

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

