# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from chromadb.api.models.Collection import GetResult
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


ChromaMetaType = Union[str, int, float, bool]


def _check_ids_and_documents(ids: Optional[List[str]], documents: List[Document]) -> Optional[List[str]]:
    if ids is None or len(ids) == 0:
        return None

    assert len(ids) == len(documents), f"{len(ids)} ids provided with {len(documents)} documents!"
    return ids


def _documents_match(docs: List[Document], ids: Optional[List[str]], vector_store: Chroma) -> bool:
    if vector_store._collection.count() != len(docs):
        print(
            "[ChromaDB Loading Check] Document quantity not matched! "
            f"{vector_store._collection.count()} in store but {len(docs)} provided."
        )
        return False

    for idx in np.random.choice(len(docs), 3):
        content_in_doc: str = docs[idx].page_content
        meta_in_doc: dict = docs[idx].metadata
        if ids is not None:
            res = vector_store.get(ids=ids[idx])
            if len(res) == 0 or len(res["documents"]) == 0:
                print(f"[ChromaDB Loading Check] No data with id {ids[idx]} exist!")
                return False
            content_in_store = res["documents"][0]
            meta_in_store =res["metadatas"][0]
        else:
            doc_in_store = vector_store.similarity_search(query=content_in_doc, k=1)[0]
            content_in_store = doc_in_store.page_content
            meta_in_store = doc_in_store.metadata

        if content_in_store != content_in_doc:
            print(
                "[ChromaDB Loading Check] Document Content not matched:\n"
                f"  In store: {content_in_store}\n"
                f"  In Doc: {content_in_doc}"
            )
            return False

        for key, value in meta_in_doc.items():
            if key not in meta_in_store:
                print(f"[ChromaDB Loading Check] Metadata {key} in doc but not in store!")
                return False

            if isinstance(value, float):
                if abs(value - meta_in_store[key]) > 1e-9:
                    print(f"[ChromaDB Loading Check] Metadata {key} not matched: {value} v.s. {meta_in_store[key]}")
                    return False
            elif meta_in_store[key] != value:
                print(f"[ChromaDB Loading Check] Metadata {key} not matched: {value} v.s. {meta_in_store[key]}")
                return False

    return True


def load_vector_store(
    collection_name: str,
    persist_directory: str,
    embedding: Embeddings=None,
    documents: List[Document]=None,
    ids: List[str]=None,
    exist_ok: bool=True,
    metadata: dict=None,
) -> Chroma:
    vector_store = Chroma(collection_name, embedding, persist_directory, collection_metadata=metadata)

    if documents is None or len(documents) == 0:
        return vector_store

    assert exist_ok or vector_store._collection.count() == 0, f"Collection {collection_name} already exist!"

    ids = _check_ids_and_documents(ids, documents)

    if _documents_match(documents, ids, vector_store):
        print(f"Chroma DB: {collection_name} loaded.")
        return vector_store

    vector_store.delete_collection()

    # Direct using of vector_store.add_documents() will raise InvalidCollectionException.
    print(f"Start to build up the Chroma DB: {collection_name}")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_metadata=metadata,
    )
    print(f"Chroma DB: {collection_name} Building-Up finished.")
    return vector_store


class ChromaMixin:
    def _init_chroma_mixin(self):
        self.retrieve_k: int = self._retriever_config.get("retrieve_k", 4)
        self.retrieve_score_threshold: float = self._retriever_config.get("retrieve_score_threshold", 0.5)

    def _get_doc_with_query(
        self, query: str, store: Chroma, retrieve_k: int=None, score_threshold: float=None,
    ) -> List[Tuple[Document, float]]:
        """Using the given `query` to query documents from the given vector store `store`.

        Returns:
            List[Tuple[Document, float]]: each item is a pair of (document, relevance score).
        """
        if retrieve_k is None:
            retrieve_k = self.retrieve_k
        if score_threshold is None:
            score_threshold = self.retrieve_score_threshold

        infos: List[Tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query=query,
            k=retrieve_k,
            score_threshold=score_threshold,
        )

        filtered_docs = [(doc, score) for doc, score in infos if score >= score_threshold]
        sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)

        return sorted_docs

    def _get_infos_with_given_meta(
        self, store: Chroma, meta_name: str, meta_value: Union[ChromaMetaType, List[ChromaMetaType]],
    ) -> Tuple[List[str], List[str], List[Dict[str, ChromaMetaType]]]:
        """Get document info in given `store` with metadata `meta_name` in given value / value list `meta_value`.

        Returns:
            List[str]: the ids of documents meet the condition.
            List[str]: the page contents of documents meet the condition.
            List[Dict[str, BasicMetaType]]: the metadata dict of documents meet the condition.
        """
        if isinstance(meta_value, list):
            filter = {meta_name: {"$in": meta_value}}
        else:
            filter = {meta_name: meta_value}

        results: GetResult = store.get(where=filter)
        ids, chunks, metadatas = results["ids"], results["documents"], results["metadatas"]
        return ids, chunks, metadatas

    def _get_scoring_func(self, store: Chroma):
        return store._select_relevance_score_fn()
