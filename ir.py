import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

with open('sample.json', 'r') as file:
    data = json.load(file)

corpus = [item['chapter_text'] for item in data]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
X = normalize(X, axis=1, norm="l1")

query = ["Why is the womb warm?"]
query_vector = vectorizer.transform(query)
query_vector = normalize(query_vector, axis=1, norm="l1")

# Now the multiplication
res = np.matmul(X.toarray(), np.transpose(query_vector.toarray()))

res_data = data[np.argmax(res)]
print(res_data)