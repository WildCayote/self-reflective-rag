from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.core.openapi.inference.models import EmbeddingsList
import os
from dotenv import load_dotenv, find_dotenv

class PineconeEmbeddingManager:
    def __init__(self, api_key: str, index_name: str, name_space: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.name_space = name_space
    
    def create_embeddings(self, documents: List[Document], model_name: str = "llama-text-embed-v2") -> EmbeddingsList:
        return self.pc.inference.embed(
            model=model_name,
            inputs=[d.page_content for d in documents],
            parameters={"input_type": "passage", "truncate": "END"}
        )
    
    def store_embeddings(self, embeddings: EmbeddingsList, documents: List[Document]):
        index = self.pc.Index(name=self.index_name)
        records = [
            {"id": f"vector{idx}", "values": e['values'], "metadata": {"text": d.page_content}}
            for idx, (d, e) in enumerate(zip(documents, embeddings))
        ]
        return index.upsert(vectors=records, namespace=self.name_space)
    
    def create_and_store_embeddings(self, documents: List[Document]):
        embeddings = self.create_embeddings(documents)
        result = self.store_embeddings(embeddings, documents)
        print(result)
    
    def search_matching(self, query: str, model: str = "llama-text-embed-v2", top_k: int = 3):
        index = self.pc.Index(name=self.index_name)
        query_embedding = self.pc.inference.embed(
            model=model,
            inputs=[query],
            parameters={"input_type": "query"}
        )


        results =  index.query(
            namespace=self.name_space,
            vector=query_embedding[0].values,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )['matches']

        documents = [result['metadata']['text'] for result in results]

        return documents
        
if __name__ == '__main__':
    load_dotenv(find_dotenv())
    api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index = os.environ.get('INDEX_NAME')
    pinecone_namespace = os.environ.get('NAMESPACE')

    manager = PineconeEmbeddingManager(api_key=api_key, index_name='kifiya', name_space='test')
    
    markdown_document = '''
    ### This is a test section
    Here is the data.

    #### This is a test sub-section
    Here is the data.
    '''

    headers_to_split_on = [("###", "Header 3"), ("####", "Header 4")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    
    manager.create_and_store_embeddings(md_header_splits, index_name=pinecone_index, name_space=pinecone_namespace)
    
    result = manager.search_matching("Where can I contact kifiya?")
    for doc in result:
        print(doc)
