---
title: Numpy RAG powered assistant
feed: show
date: 2026-09-03
---

In this project, I wanted to take a very small LLM, sub 1B parameters, and boost its performance with a RAG system. I chose to work in the context of the **NumPy** documentation, since it contains many details, arguments and functions, so a small LLM is prone to hallucinating about it.

The LLM I decided to use is [`ibm-granite/granite-4.0-h-350m`](https://huggingface.co/ibm-granite/granite-4.0-h-350m), which has only 350M parameters. It is so lightweight that it runs on my machine at incredible speed, and it also supports parallel calls on a single `ollama serve`. Because the model is so small, it does not hold much knowledge, which makes the difference between the plain chat and the RAG-powered chat more noticeable. For a production project one should, of course, use the best LLM available.

The RAG system I used is the default [`chromadb`](https://www.trychroma.com/home) with the default sentence transformer `sentence-transformers/all-MiniLM-L6-v2`.

To keep the project *scalable*, everything was set up with Docker containers (frontend, backend, ollama) and brought up together with Docker Compose. The frontend and backend communicate through a REST API built with `fastapi`. The frontend was built with `streamlit`. The LLM can be served either by an `ollama` instance running in a Docker container or by the `ollama` server on the host (on macOS, Docker cannot use Metal inside a container); we could also use a cloud LLM provider such as ChatGPT or Claude, but we chose not to so that everything runs locally.

# Getting the context

To build the RAG system, we need extra information to provide as context to the LLM. For that, I downloaded the entire [**NumPy 2.5** documentation](https://numpy.org/doc/2.5/numpy-html.zip), extracted it and parsed it with the **BeautifulSoup** package. For each HTML file (i.e. each page), I treated every **\<article\>** HTML element as one chunk available for matching. There are other ways of doing this. In general one should try several chunk sizes, with or without overlap, but in my case this did not make much sense: each page is independent of the others, and it would make no sense to split a single page. For this reason, each HTML page is its own chunk.

To build the RAG system, I used `chromadb`:

``` python
import chromadb

chroma_client = chromadb.PersistentClient("../backend/chroma.db")

collection = chroma_client.get_or_create_collection(name="numpy_docs")
collection.add(ids=ids, documents=documents, metadatas=metadatas)
```

This creates a database with the embeddings of the NumPy docs. We can then query with another string and check whether any similar or relevant sentences appear in the docs:

``` python
collection.query(query_texts=["How to reshape an numpy array?"])
```

For this example, the 5th result starts with `numpy.ndarray.reshape#\nmethod\nndarray.reshape(shape, /, *, order='C', copy=None)...`, which is exactly the [function to use](https://numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html).

Under the hood, this uses a neural network (`all-MiniLM-L6-v2`) to project each chunk of text into a $384$-dimensional space that encapsulates, in some way, the meaning of the chunk. We do this for every chunk created while parsing the documentation, building a vector database that holds both the embedding and the text; this is done for performance, as the embedding of the documentation is computed only once. When we want to query for similar sentences, we embed the query and find which chunks of the vector database have the most similar embedding using the cosine distance, i.e. the "distance" is given by
$$d_{q,v} = 1-\frac{\vec{e}_v \cdot \vec{e}_q}{|\vec{e}_v| |\vec{e}_q|}$$
where $\vec{e}_v$ is the embedding of each chunk in the vector database and $\vec{e}_q$ is the embedding of the query. For this project, we add to the LLM context the 10 chunks with the lowest $d_{q,v}$. This number can be tuned per project. If the RAG system fails to retrieve any relevant chunk, the LLM is instructed not to answer, although sometimes it does so anyway.

Additional layers can be added to the stack, such as a reranker that inspects the retrieved chunks and picks the most relevant ones. Rerankers are considerably slower than the sentence transformers that typical RAG systems use, but they are fast enough to run on the top-$N$ results of the RAG. A reranker was beyond the scope of this project.

# Backend

The backend was built with `fastapi` and `chromadb`, and was designed to work asynchronously, i.e. it can handle multiple requests at the same time. I exposed a POST endpoint that takes as input the class

``` python
class Query(BaseModel):
    messages: list = []
    rag: str
```

where `messages` is the list of messages and `rag` is a boolean that determines whether the chat will be RAG-powered or not. Only the first chat message is passed to the RAG model to provide context. Follow-up messages are answered using only the available context, i.e. the conversation plus the initial RAG context.

# Frontend

The frontend was built with `streamlit`, based on an [AI chat Streamlit example](https://github.com/streamlit/demo-ai-assistant). The page starts with a single input textbox that queries both LLMs, and we provide a few suggestions of questions to ask.

![[36a03ffd5fedf4e3744cec52a2475ea854d4a8a0.png]]

After the first query, each LLM answers the question, one using only its own knowledge and the other using the RAG context. The page then splits into two, and the user can continue the conversation with either chat. Two buttons are provided: the `Restart` button resets both chats, and the `Show context` button shows which context was given to the LLM, which can help troubleshoot issues with the models.

# Docker

I also designed this project to showcase my ability to build a production-style deployment. For that reason, the project is split into three containers:

- Ollama container - runs the LLM.
- Backend container - contains `fastapi` and `chromadb`.
- Frontend container - hosts the `streamlit` frontend.

The backend and frontend interact only over an HTTP REST API. The backend and the ollama container interact through the `ollama` package, which also uses a REST API under the hood. The frontend and the ollama container do not interact at all.

To orchestrate everything we use `docker compose`; the YAML file is

``` yml
services:
  ollama:
    image: granite4_350m:latest
  frontend:
    image: numpyragfrontend:latest
    ports:
      - "8080:8080"
  backend:
    image: numpyragbackend:latest
```

where only port `8080` is published on the host machine, so that the backend and ollama are completely isolated from external interference.

# Results

In this first image I ask what the options are for the `density` flag of the NumPy histogram, which can be either `True` or `False`. The RAG-powered chat gives the correct answer while the plain chat hallucinates.

![[cd8bba5f342bdd6ac63edec6d727220fdc0e7478.png]]

In this second example, we can see the knowledge limitations of the small LLM: it does not know that Chebyshev polynomial fitting is already implemented in NumPy, and instead provides a much more cumbersome answer.

![[bc7f3fbd8a7421ebd6b2f7b455b76548d1aab922.png]]

In the third example, the LLM refuses to answer the question because it is beyond the scope of its knowledge.

![[4c5cc6b249c07770b600fd77ae28455fece6d25f.png]]

# Conclusions

This RAG system is, of course, overkill: a decent-sized LLM most likely already knows enough to answer these NumPy questions. Nevertheless, similar implementations can be used for internal company documentation, customer support chatbots, and so on.
