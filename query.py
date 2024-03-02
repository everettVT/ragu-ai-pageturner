import json
import typing

from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections
from langchain.chains import RetrievalQA

import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

milvus_endpoint = config['milvus']['host']  #'localhost' #'https://in03-edeea0a7bd3937f.api.gcp-us-west1.zillizcloud.com'
port = config['milvus']['port']
connection_args = {"host": milvus_endpoint, "port": f'{port}'}
milvus_address=f'{milvus_endpoint}:{port}'

milvus_endpoint = 'localhost' #'https://in03-edeea0a7bd3937f.api.gcp-us-west1.zillizcloud.com'
connection_args = {"host": milvus_endpoint, "port": '19530'}
milvus_address=f'{milvus_endpoint}:19530'


openai_key = config['openai']['key']


def load_db(collection_name: str, embeddings, connection_args: typing.Mapping[str, str]):
    vector_db = Milvus(
        embeddings,
        connection_args=connection_args,
        collection_name=collection_name,
    )
    return vector_db


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
    # collection_name = sys.argv[1]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    collection_name = f"demo_collection"
    query = sys.argv[1]
    print(query_db(collection_name, query, embeddings))



if __name__ == '__main__':
    main()
