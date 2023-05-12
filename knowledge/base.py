import os
from glob import glob

import torch
from langchain import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

content_folder = 'data/content'
knowledge_folder = 'data/knowledge'
embedding_model = 'GanymedeNil/text2vec-large-chinese'
embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'

embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': embedding_device})

chunk_size = 200
chunk_overlap = 50


def list_knowledge():
    paths = glob(f'{knowledge_folder}/*')
    knowledge_list = [os.path.basename(path) for path in paths]
    knowledge_list.sort()
    return knowledge_list


def dump_knowledge(name, content):
    content_path = f'{content_folder}/{name}.txt'
    with open(content_path, mode='w', encoding='utf-8') as file:
        file.write(content)

    loader = TextLoader(f'{content_folder}/{name}.txt', encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(f'{knowledge_folder}/{name}')


def load_knowledge(name):
    vs = FAISS.load_local(f'{knowledge_folder}/{name}', embeddings)
    return vs


def load_knowledge_content(name):
    content_path = f'{content_folder}/{name}.txt'
    with open(content_path, mode='r', encoding='utf-8') as file:
        return file.read()


def query_knowledge(name, q, topk=1):
    vs = FAISS.load_local(f'{knowledge_folder}/{name}', embeddings)
    docs_and_scores = vs.similarity_search_with_score(q, k=topk)
    return docs_and_scores
