import json
from sklearn.feature_extraction.text import TfidfVectorizer

with open('sample.json', 'r') as file:
    data = json.load(file)

corpus = [item['chapter_text'] for item in data]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
print(data)