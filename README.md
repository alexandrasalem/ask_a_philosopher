This is the code for developing the **Ask a Philosopher** chatbot. 
We started with building one based upon the text of Aristotle.
This project uses retrieval-augmented generation (RAG) for answering questions, based up recommendations from Jurafsky \& Martin's book, *Speech and Language Processing*, 3rd edition (https://web.stanford.edu/~jurafsky/slp3/14.pdf).

In RAG, first a relevant document to the query is identified using standard Information Retrieval (IR) techniques. Then, a generative LLM generates an answer given the query and retrieved document.

We use Python 3.11.9. Install requirements using `pip install -r requirements.txt`.

Details on our code thus far (WIP):

We pulled down the Aristotle chapters with `pull_data.py`.

Then we made the json file with `making_json.py`.

Tools for the basic IR are in `ir.py`.

The script `version_1_basic_ir.py` generates the closest document to the query and returns that document as a string.

The script `version_1_basic_ir_cos_sims.py` pulls cosine similarity values between the query and each of the documents, and returns this as a list of dictionaries.

The script `version_1_basic_ir_multiple_questions.py` generates the closest documents to a collection of queries and returns those as a string.

The script `version_1_basic_ir_user_input.py` generates the closest document to a query provided by the user in streamlit and returns that document as a string.

Each of the scripts above uses a basic IR system with tf-idf. Variations of each of these scripts for version 2 which uses BERT instead of tf-idf for the IR are now available (`version_2_bert_ir.py`, etc).