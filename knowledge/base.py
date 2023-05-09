import os
from glob import glob

import torch
from langchain import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

content_folder = 'data/content'
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}
embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict['text2vec'],
                                   model_kwargs={'device': embedding_device})


def list_knowledge():
    paths = glob(f'{content_folder}/*')
    knowledge_list = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    knowledge_list.sort()
    return knowledge_list


def load_knowledge(name):
    with open(f'{content_folder}/{name}.txt', mode='r', encoding='utf-8') as file:
        content = file.read()
    return content


def dump_knowledge(name, content):
    with open(f'{content_folder}/{name}.txt', mode='w', encoding='utf-8') as file:
        file.write(content)


def query_knowledge(name, q, topk=1):
    loader = TextLoader(f'{content_folder}/{name}.txt', encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    docs_and_scores = db.similarity_search_with_score(q, k=topk)
    return docs_and_scores
