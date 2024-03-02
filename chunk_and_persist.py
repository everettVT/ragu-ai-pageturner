import json
import os
import time
import typing

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from pymilvus import connections, utility
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain.chains import RetrievalQA

from milvus import MilvusServer, MilvusServerConfig

import yaml


with open("config.yaml") as f:
    config = yaml.safe_load(f)

milvus_endpoint = config['milvus']['host']  #'localhost' #'https://in03-edeea0a7bd3937f.api.gcp-us-west1.zillizcloud.com'
port = config['milvus']['port']
connection_args = {"host": milvus_endpoint, "port": f'{port}'}
milvus_address=f'{milvus_endpoint}:{port}'

OPENAI_API_KEY_PATH='../apikeys.json'
openai_key = config['openai']['key'] #json.load(open(OPENAI_API_KEY_PATH))["key"]

def chunk_and_persist(collection_name: str, docs_list: typing.List[Document], connection_args: typing.Mapping[str, str], embeddings):
    return Milvus.from_documents(
        documents=docs_list,
        embedding=embeddings,
        # drop_old=True,
        connection_args=connection_args,
        collection_name=collection_name)


def drop_collection(collection_name: str):
    connections.connect(alias="del", address=milvus_address)
    # connections.connect("del",
    #                 uri=milvus_endpoint,
    #                 token=zilliz_api_key)
    utility.drop_collection(collection_name, using="del")


def chunk_text(docs: typing.List[Document], chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


def load_db(collection_name: str, embeddings, connection_args: typing.Mapping[str, str]):
    vector_db = Milvus(
        embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
    )
    return vector_db



def index_docs_at_path(collection_name: str, dir_path: str, embeddings):
    loader = DirectoryLoader(dir_path, glob="**/*.txt")
    docs = loader.load()
    index_docs(collection_name, docs, embeddings)


def index_docs(collection_name: str, docs: typing.List[Document], embeddings):
    chunked_docs = chunk_text(docs, chunk_size=512, chunk_overlap=0)
    chunk_and_persist(collection_name, chunked_docs, connection_args=connection_args, embeddings=embeddings)


def query_db(collection_name: str, query: str, embeddings):
    vector_db = load_db(collection_name, embeddings, connection_args)
    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={'k': 5}),
        verbose=True,
        chain_type_kwargs={
            "verbose": True
        })
    return qa.invoke(query)


def main():
    import sys
    collection_name = sys.argv[1]
    doc_path =sys.argv[2]
    # drop_collection(collection_name)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    file_names = os.listdir(doc_path)
    docs = []
    for f_name in file_names:
        with open(os.path.join(doc_path, f_name), mode='rt', encoding='utf8') as f:
            print(f"Reading {f_name}")
            doc_contents = f.read()
            docs.append(Document(page_content=doc_contents))
    index_docs(collection_name, docs, embeddings)
    print("Indexed")


if __name__ == '__main__':
    main()
