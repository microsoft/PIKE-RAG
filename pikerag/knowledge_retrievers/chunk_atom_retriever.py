# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pikerag.knowledge_retrievers.base_qa_retriever import BaseQaRetriever
from pikerag.knowledge_retrievers.mixins.chroma_mixin import ChromaMixin, load_vector_store
from pikerag.utils.config_loader import load_callable, load_embedding_func
from pikerag.utils.logger import Logger


@dataclass
class AtomRetrievalInfo:
    atom_query: str
    atom: str
    source_chunk_title: str
    source_chunk: str
    source_chunk_id: str
    retrieval_score: float
    atom_embedding: List[float]


class ChunkAtomRetriever(BaseQaRetriever, ChromaMixin):
    """A retriever contains two vector storage and supports several retrieval method.

    There are two Vector Stores inside this retriever:
    - `_chunk_store`: The one for chunk storage.
    - `_atom_store`: The one for atom storage. Each atom doc in the this storage is linked to a chunk in `_chunk_store`
        by the metadata named `source_chunk_id`.

    There are four public interface to retrieve information by this retriever:
    - `retrieve_atom_info_through_atom`: to retrieve atom info through atom storage by queries
    - `retrieve_atom_info_through_chunk`: to retrieve atom info through chunk storage by query
    - `retrieve_contents_by_query`: to retrieve chunk contents through both atom storage and chunk storage
    - `retrieve_contents`: equal to `retrieve_contents_by_query(query=qa.question)`
    """
    name: str = "ChunkAtomRetriever"

    def __init__(self, retriever_config: dict, log_dir: str, main_logger: Logger) -> None:
        super().__init__(retriever_config, log_dir, main_logger)

        self._load_vector_store()

        self._init_chroma_mixin()

        self.atom_retrieve_k: int = retriever_config.get("atom_retrieve_k", self.retrieve_k)

    def _load_vector_store(self) -> None:
        assert "vector_store" in self._retriever_config, "vector_store must be defined in retriever part!"
        vector_store_config = self._retriever_config["vector_store"]

        collection_name = vector_store_config.get("collection_name", self.name)
        doc_collection_name = vector_store_config.get("collection_name_doc", f"{collection_name}_doc")
        atom_collection_name = vector_store_config.get("collection_name_atom", f"{collection_name}_atom")

        persist_directory = vector_store_config.get("persist_directory", None)
        if persist_directory is None:
            persist_directory = self._log_dir
        exist_ok = vector_store_config.get("exist_ok", True)

        embedding_config = vector_store_config.get("embedding_setting", {})
        self.embedding_func: Embeddings = load_embedding_func(
            module_path=embedding_config.get("module_path", None),
            class_name=embedding_config.get("class_name", None),
            **embedding_config.get("args", {}),
        )

        self.similarity_func = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        print(f"\n🔧 [ChunkAtomRetriever] Starting vector store initialization...")
        print(f"📁 Storage directory: {persist_directory}")
        print(f"🎯 Embedding model: {embedding_config.get('class_name', 'Default')}")
        
        # 加载并创建 Chunk Store
        print(f"\n📚 [Step 1/2] Loading document chunks...")
        loading_configs = vector_store_config["id_document_loading"]
        doc_ids, docs = load_callable(
            module_path=loading_configs["module_path"],
            name=loading_configs["func_name"],
        )(**loading_configs.get("args", {}))
        
        print(f"📊 Loaded {len(docs)} document chunks")
        print(f"🏗️  Creating chunk vector store: {doc_collection_name}")
        
        self._chunk_store: Chroma = load_vector_store(
            collection_name=doc_collection_name,
            persist_directory=persist_directory,
            embedding=self.embedding_func,
            documents=docs,
            ids=doc_ids,
            exist_ok=exist_ok,
        )

        # 加载并创建 Atom Store
        print(f"\n⚛️  [Step 2/2] Loading atomic questions...")
        loading_configs = vector_store_config["id_atom_loading"]
        atom_ids, atoms = load_callable(
            module_path=loading_configs["module_path"],
            name=loading_configs["func_name"],
        )(**loading_configs.get("args", {}))
        
        print(f"📊 Loaded {len(atoms)} atomic questions")
        print(f"🏗️  Creating atom vector store: {atom_collection_name}")
        
        self._atom_store: Chroma = load_vector_store(
            collection_name=atom_collection_name,
            persist_directory=persist_directory,
            embedding=self.embedding_func,
            documents=atoms,
            ids=atom_ids,
            exist_ok=exist_ok,
        )
        
        print(f"\n✅ [ChunkAtomRetriever] Vector store initialization completed!")
        print(f"📈 Total collections created: 2 (chunks + atoms)")
        print(f"💾 Data persisted to: {persist_directory}")

    def _atom_info_tuple_to_class(self, atom_retrieval_info: List[Tuple[str, Document, float]]) -> List[AtomRetrievalInfo]:
        # Extract all unique `source_chunk_id`
        source_chunk_ids: List[str] = list(set([doc.metadata["source_chunk_id"] for _, doc, _ in atom_retrieval_info]))

        # Retrieve corresponding source chunks and formulate as an id2chunk dict.
        chunk_doc_results: Dict[str, Any] = self._chunk_store.get(ids=source_chunk_ids)
        chunk_id_to_content = {
            chunk_id: chunk_str
            for chunk_id, chunk_str in zip(chunk_doc_results["ids"], chunk_doc_results["documents"])
        }

        # Wrap up.
        retrieval_infos: List[AtomRetrievalInfo] = []
        for atom_query, atom_doc, score in atom_retrieval_info:
            source_chunk_id = atom_doc.metadata["source_chunk_id"]
            retrieval_infos.append(
                AtomRetrievalInfo(
                    atom_query=atom_query,
                    atom=atom_doc.page_content,
                    source_chunk_title=atom_doc.metadata.get("title", None),
                    source_chunk=chunk_id_to_content[source_chunk_id],
                    source_chunk_id=source_chunk_id,
                    retrieval_score=score,
                    atom_embedding=self.embedding_func.embed_query(atom_doc.page_content),
                )
            )

        return retrieval_infos

    def retrieve_atom_info_through_atom(
        self, queries: Union[List[str], str], retrieve_id: str="", **kwargs,
    ) -> List[AtomRetrievalInfo]:
        """Retrieve the relevant atom and its source chunk by the given atom queries.

        Args:
            atom_queries (Union[List[str], str]): A list of queries that would be used to query the `_atom_store`.
            retrieve_id (str): id to identifying the query, could be used in logging.

        Returns:
            List[AtomRetrievalInfo]: The retrieved atom information would be returned together with its corresponding
                source chunk information.
        """
        # Decide which retrieve_k to use.
        if "retrieve_k" in kwargs:
            retrieve_k: int = kwargs["retrieve_k"]
        elif isinstance(queries, list) and len(queries) > 1:
            retrieve_k: int = self.atom_retrieve_k
        else:
            retrieve_k: int = self.retrieve_k

        # Wrap atom_queries into a list if only one element given.
        if isinstance(queries, str):
            queries = [queries]

        # Query `_atom_store` to get relevant atom information.
        query_atom_score_tuples: List[Tuple[str, Document, float]] = []
        for atom_query in queries:
            for atom_doc, score in self._get_doc_with_query(atom_query, self._atom_store, retrieve_k):
                query_atom_score_tuples.append((atom_query, atom_doc, score))

        # Wrap to predefined dataclass.
        return self._atom_info_tuple_to_class(query_atom_score_tuples)

    def _chunk_info_tuple_to_class(self, query: str, chunk_docs: List[Document]) -> List[AtomRetrievalInfo]:
        # Calculate the best-hit (atom, similarity score, atom embedding) for each chunk.
        best_hit_atom_infos: List[Tuple[str, float, List[float]]] = []
        query_embedding = self.embedding_func.embed_query(query)
        for chunk_doc in chunk_docs:
            best_atom, best_score, best_embedding = "", 0, []
            for atom in chunk_doc.metadata["atom_questions_str"].split("\n"):  # TODO
                atom_embedding = self.embedding_func.embed_query(atom)
                score = self.similarity_func(query_embedding, atom_embedding)
                if score > best_score:
                    best_atom, best_score, best_embedding = atom, score, atom_embedding
            best_hit_atom_infos.append((best_atom, best_score, best_embedding))

        # Wrap up.
        retrieval_infos: List[AtomRetrievalInfo] = []
        for chunk_doc, (atom, score, atom_embedding) in zip(chunk_docs, best_hit_atom_infos):
            retrieval_infos.append(
                AtomRetrievalInfo(
                    atom_query=query,
                    atom=atom,
                    source_chunk_title=chunk_doc.metadata.get("title", None),
                    source_chunk=chunk_doc.page_content,
                    source_chunk_id=chunk_doc.metadata["id"],
                    retrieval_score=score,
                    atom_embedding=atom_embedding,
                )
            )
        return retrieval_infos

    def retrieve_atom_info_through_chunk(self, query: str, retrieve_id: str="") -> List[AtomRetrievalInfo]:
        """Retrieve the relevant chunk and its atom with best hit by the given query.

        Args:
            query (str): A query that would be used to query the `_chunk_store`.
            retrieve_id (str): id to identifying the query, could be used in logging.

        Returns:
            List[AtomRetrievalInfo]: The retrieved chunk information would be returned together with its best-hit atom
                information.
        """
        # Query `_chunk_store` to get relevant chunk information.
        chunk_info: List[Tuple[Document, float]] = self._get_doc_with_query(query, self._chunk_store, self.retrieve_k)

        # Wrap to predefined dataclass.
        return self._chunk_info_tuple_to_class(query=query, chunk_docs=[doc for doc, _ in chunk_info])

    def retrieve_contents_by_query(self, query: str, retrieve_id: str="") -> List[str]:
        """Retrieve the relevant chunk contents by the given query. The given query would be used to query both
        `_atom_store` and `_chunk_store`.

        Args:
            query (str): A query that would be used to query the vector stores.
            retrieve_id (str): id to identifying the query, could be used in logging.

        Returns:
            List[str]: The retrieved relevant chunk contents, including two kinds of chunks: the chunk retrieved
                directly from the `_chunk_store` and the corresponding source chunk linked by the atom retrieved from
                the `_atom_store`.
        """
        # Retrieve from `_chunk_store` by query to get relevant chunk directly.
        chunk_info: List[Tuple[Document, float]] = self._get_doc_with_query(query, self._chunk_store, self.retrieve_k)
        chunks = [chunk_doc.page_content for chunk_doc, _ in chunk_info]

        # Retrieve through `_atom_store` and get relevant source chunk.
        atom_infos = self.retrieve_atom_info_through_atom(queries=query, retrieve_id=retrieve_id)
        atom_source_chunks = [atom_info.source_chunk for atom_info in atom_infos]

        # Add unique source chunk to `chunks`.
        for chunk in atom_source_chunks:
            if chunk not in chunks:
                chunks.append(chunk)
        return chunks
