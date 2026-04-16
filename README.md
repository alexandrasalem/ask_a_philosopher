# Ask a Philosopher

### Description
This is the code repository for building our *Ask a Philosopher* chatbot for chatting with either AI-Aristotle or AI-Confucius.
The chatbot uses Retrieval-Augmented Generation (RAG) to answer user queries. 
For *retrieval*, we use Octen-Embedding-0.6B, a text embedding model for dense embeddings, for finding the closest philosophical texts to the queries to use as reference.
For *generation*, we use Llama-3.2-3B-Instruct, an open-source, instruction-tuned LLM from Meta.
We only use the retrieved references when the cosine similarity beats a specified threshold for each philosopher.
Both retrieval and generation steps are run through a serverless RunPod instance for GPU usage.

### RunPod set-up
The RunPod code is set up with `Dockerfile` including the following:
* `llm.py`:  LLM generation of responses
* `ir_light.py`: IR setup with octen
* `rp_handler_ask_a_phil.py`:  runpod handler file
* `requirements.txt`:  required install for runpod
* `aristotle.json`: complied aristotle chapters 
* `confucius.json`: compiled confucius chapters 
* `aristotle_octen_small.csv`:  pre-computed embeddings for aristotle chapters
* `confucius_octen_small.csv`:  pre-computed embeddings for confucius chapters

### Other files
Other files:
* `pull_data.py`: script for pulling down the chapters
* `making_json.py`: script for creating the json file
* `create_bert_files.py`: script for pre-computing the embeddings

### Deployed Streamlit app
The repository for the deployed application is here: https://github.com/alexandrasalem/ask_a_philosopher_streamlit. 
The app is deployed through Render, using Streamlit for the infrastructure.
It also incorporates one of Google's English neural Text-to-Speech systems for the philosopher responses.
The chatbot is deployed at this url: https://ask-a-philosopher.onrender.com/ 
A password is required. 