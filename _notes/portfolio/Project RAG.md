---
title: Numpy RAG powered assistant
feed: show
date: 2026-09-03
---
# Project RAG

In this project, I wanted to use a very small LLM, sub 1b parameters, to build a **numpy** RAG powered assistance. 

The LLM I decided to use is `ibm-granite/granite-4.0-h-350m` (https://huggingface.co/ibm-granite/granite-4.0-h-350m) which has only 350m parameters. This LLM is so lightweight that can run on my system at incredible speeds, and also supports parallel calls on the same `ollama serve`. This LLM is so small that it does not have much knowledge, this will make the difference between the normal chat and RAG chat more noticeable. Of course, for a production project one should use the best LLM available.

The RAG system I used was the default `chromadb` (https://www.trychroma.com/home) using the default sentence transformer `sentence-transformers/all-MiniLM-L6-v2`.

In order for the project to be *scalable*, everything was setup using docker containers (frontend, backend, ollama) and simultaneous brought to life using docker compose. The frontend and backend containers interact using REST API which was create using `fastapi`. The frontend was designed using `streamlit`. The LLM can be provided by `ollama` running on a Docker container, or on the host `ollama` server (for macos Docker cannot use metal inside Docker container); we could also use a cloud LLM provider, such as ChatGPT or Claude, but we chose not to so that everything runs locally.

# Get the context

To build the RAG system, one needs additional information to provide as context to the LLM. For that, I download the entire **numpy 2.5** documentation (https://numpy.org/doc/2.5/numpy-html.zip). Extracted it and parsed it using the **BeautifulSoup** package. For each HTML file (for each page), I considered each **\<article\>** HTML object to be one chunck available for matching. There are other ways of doing this. In general one should try some chunking sizes with/without overlap bu, for my case this did not make much sense as each page is independent of other pages; and it would make no sense to split one of these pages. For this reason, each HTML page has its one chunck.

To build the RAG system, I used `chromabd`

``` python
import chromadb

chroma_client = chromadb.PersistentClient("../backend/chroma.db")

collection = chroma_client.get_or_create_collection(name="numpy_docs")
collection.add(ids=ids, documents=documents, metadatas=metadatas)
```

This command creates a database with the embedding of the numpy docs. Then we can query other string and check if any similar/relevant sentences appear in the docs using this command

``` python
collection.query(query_texts=["How to reshape an numpy array?"])
```

For this example, the 5th results starts with `numpy.ndarray.reshape#\nmethod\nndarray.reshape(shape, /, *, order='C', copy=None)...` which is the correct function to be used (https://numpy.org/devdocs/reference/generated/numpy.ndarray.reshape.html).

Under the hood, this uses a neural network (`all-MiniLM-L6-v2`) to project each chunck of text into a $384$ dimensional space which will encapsulate, in some way, the meaning of the chunck of text. We do this to all chuncks created during the parsing of the documentation to build a vector database, where we have the embedding and the text, this is done to increase performance as we only calculate the embedding of the documentation once. When we want to query for similar sentences, we embed our query and find which chuncks of the vector database have the most similar embedding using the cossine distance, i.e. the "distance" is given by
$$d_{q,v} = 1-\frac{\vec{e}_v \cdot \vec{e}_q}{|\vec{e}_v| |\vec{e}_q|}$$
where $\vec{e}_v$ is the embedding of each chunck in the vector database and $\vec{e}_q$ is the embedding of the query. For this project, add to the LLM context the top 10 embedding with the lowest $d_{q,v}$. This number can be fine-tuned for each project. If the RAG system failed to retrieve any relevant chuncks, the LLM was instructed not to answer, although sometimes it fails to do so.

Additional layers to the stack can be added, such as a reranker that looks at the retrieved chuncks and looks for the most relevant ones. These rerankers are considerable slower than these sentence transformers that typical RAG systems use, but are fast enough to be used on the $N$-top results of the RAG. A reranker was beyond the scope of this project.

# Backend

The backend was built using `fastapi` and `chromadb`, as was designed to work asynchronously, i.e. it can handle multiple requests at the same time. I opened POST method that takes as input a class

``` python
class Query(BaseModel):
    messages: list = []
    rag: str
```

where `messages` is the list of messages and `rag` is a boolean that determines if the chat will RAG powered or not. I selected that only first chat message can be passed to the RAG model to give some context. Follow up messages will be answered only using the available context, i.e. the conversation + initial RAG context.

# Frontend

The frontend was designed using `streamlit`, based on a [AI chat streamlit example](https://github.com/streamlit/demo-ai-assistant). The page starts with a single input textbox that query both LLMs, we also provide a few suggestions to ask the LLM.

<figure>
<img
src="{{ site.baseurl }}/notes/36a03ffd5fedf4e3744cec52a2475ea854d4a8a0.png"
class="wikilink" alt="Pastedimage20260903135213.png" />
<figcaption
aria-hidden="true">Pastedimage20260903135213.png</figcaption>
</figure>

After the first query, the LLM answers the question using its knowledges and using the RAG context. The page is spit into two, where the user can continue the discussing which any of the two chats. Two buttons are provided. The `Restart` button resets both chats, the `Show context` button show which context was given to the LLM, this might help troubleshoot some issues with the LLMs.
\# Docker

I designed this project also to showcase my abilities to make a production project. For that reason, this project was split into three containers

- Ollama container - Runs the LLM.
- Backend container - Contains the `fastapi` and `chromadb`.
- Frontend container - Has the `streamlit` front end

The backend and frontend interact only via HTTP REST API. The backend and the ollama container interact using the `ollama` package that uses REST API under the hood aswell. The frontend and the ollama container to not interact at all.

To orchestrate everything, we use `docker compose`, the YAML file used is

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

where we only published port `8080` on the host machine so that the backend/ollama is completely isolated from external interference.
\# Results

In this first image I ask what are the option for the `density` flag of the numpy histogram, which can either be `True` or `False`. The RAG powered chat provides the correct answer while the normal chat hallucinates.

<figure>
<img
src="{{ site.baseurl }}/notes/cd8bba5f342bdd6ac63edec6d727220fdc0e7478.png"
class="wikilink" alt="Pastedimage20260903135306.png" />
<figcaption
aria-hidden="true">Pastedimage20260903135306.png</figcaption>
</figure>

In this second example, we can see the knowledge limitations that the small LLM has, as it does not know that the Chebyshev polynomials fit is already implemented in numpy, as instead provides a much more cumbersome answer.

<figure>
<img
src="{{ site.baseurl }}/notes/bc7f3fbd8a7421ebd6b2f7b455b76548d1aab922.png"
class="wikilink" alt="Pastedimage20260903135423.png" />
<figcaption
aria-hidden="true">Pastedimage20260903135423.png</figcaption>
</figure>

In the third example, the LLM refuses to answer the questions as it is beyond the scope of its knowledge.

<figure>
<img
src="{{ site.baseurl }}/notes/4c5cc6b249c07770b600fd77ae28455fece6d25f.png"
class="wikilink" alt="Pastedimage20260903135326.png" />
<figcaption
aria-hidden="true">Pastedimage20260903135326.png</figcaption>
</figure>

# Conclusions

Of course, this RAG is an overkill as a decent sized LLM likely has the knowledge to answer these questions about numpy. Nevertheless, similar implementations can be used for internal company documentation, client assistance chat bot, etc.
