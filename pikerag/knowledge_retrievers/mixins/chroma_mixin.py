# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import os
import hashlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from chromadb.api.models.Collection import GetResult
from langchain_chroma import Chroma
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

    print(f"[ChromaDB Loading Check] Validating document consistency (checking 3 random samples)...")
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

    print(f"[ChromaDB Loading Check] Document consistency validation completed successfully!")
    return True


def _get_documents_hash(documents: List[Document]) -> str:
    """计算文档集合的hash值，用于检查数据是否变化"""
    if not documents:
        return ""
    
    # 创建文档内容和元数据的hash
    content_parts = []
    for doc in documents[:10]:  # 只检查前10个文档以提高性能
        content_parts.append(doc.page_content)
        # 添加关键元数据
        for key in sorted(doc.metadata.keys()):
            content_parts.append(f"{key}:{doc.metadata[key]}")
    
    combined_content = "|||".join(content_parts)
    return hashlib.md5(combined_content.encode('utf-8')).hexdigest()


def _check_cache_file(persist_directory: str, collection_name: str, documents_hash: str) -> bool:
    """检查缓存文件是否存在且匹配"""
    cache_file = os.path.join(persist_directory, f"{collection_name}.cache")
    
    if not os.path.exists(cache_file):
        return False
    
    try:
        with open(cache_file, 'r') as f:
            cached_hash = f.read().strip()
        return cached_hash == documents_hash
    except:
        return False


def _save_cache_file(persist_directory: str, collection_name: str, documents_hash: str):
    """保存缓存文件"""
    cache_file = os.path.join(persist_directory, f"{collection_name}.cache")
    try:
        os.makedirs(persist_directory, exist_ok=True)
        with open(cache_file, 'w') as f:
            f.write(documents_hash)
    except Exception as e:
        print(f"[ChromaDB] Warning: Failed to save cache file: {e}")


def load_vector_store(
    collection_name: str,
    persist_directory: str,
    embedding: Embeddings=None,
    documents: List[Document]=None,
    ids: List[str]=None,
    exist_ok: bool=True,
    metadata: dict=None,
) -> Chroma:
    print(f"[ChromaDB] Initializing collection: {collection_name}")
    
    if documents is None or len(documents) == 0:
        print(f"[ChromaDB] No documents provided, returning empty vector store")
        return Chroma(collection_name, embedding, persist_directory, collection_metadata=metadata)

    # 计算文档hash
    documents_hash = _get_documents_hash(documents)
    
    # 检查缓存
    if _check_cache_file(persist_directory, collection_name, documents_hash):
        print(f"[ChromaDB] ✅ Found valid cache for collection {collection_name}")
        vector_store = Chroma(collection_name, embedding, persist_directory, collection_metadata=metadata)
        
        # 快速检查collection是否存在且不为空
        if vector_store._collection.count() > 0:
            print(f"[ChromaDB] ✅ Collection {collection_name} loaded from cache ({vector_store._collection.count()} documents)")
            return vector_store
        else:
            print(f"[ChromaDB] ⚠️  Cache file exists but collection is empty, will rebuild")
    
    # 如果没有缓存或缓存无效，检查现有数据
    vector_store = Chroma(collection_name, embedding, persist_directory, collection_metadata=metadata)
    
    assert exist_ok or vector_store._collection.count() == 0, f"Collection {collection_name} already exist!"

    ids = _check_ids_and_documents(ids, documents)
    
    # 简化的文档匹配检查 - 只检查数量和随机采样1个文档
    collection_count = vector_store._collection.count()
    if collection_count == len(documents) and collection_count > 0:
        print(f"[ChromaDB] Document count matches ({collection_count}), performing quick validation...")
        try:
            # 只检查1个随机文档
            random_idx = np.random.choice(len(documents))
            content_in_doc = documents[random_idx].page_content
            
            if ids is not None:
                res = vector_store.get(ids=ids[random_idx])
                if len(res) > 0 and len(res["documents"]) > 0:
                    content_in_store = res["documents"][0]
                    if content_in_store == content_in_doc:
                        print(f"[ChromaDB] ✅ Quick validation passed, collection {collection_name} is ready")
                        _save_cache_file(persist_directory, collection_name, documents_hash)
                        return vector_store
            
        except Exception as e:
            print(f"[ChromaDB] Validation failed with error: {e}")
    
    # 需要重建collection
    print(f"[ChromaDB] 🗑️  Collection needs rebuilding...")
    if collection_count > 0:
        vector_store.delete_collection()

    # 重新构建
    print(f"[ChromaDB] 🚀 Building ChromaDB collection: {collection_name}")
    print(f"[ChromaDB] 📊 Processing {len(documents)} documents...")
    print(f"[ChromaDB] ⏳ This may take a few minutes depending on document size and embedding model...")
    
    start_time = time.time()
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_metadata=metadata,
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"[ChromaDB] ✅ Collection {collection_name} built successfully!")
    print(f"[ChromaDB] ⏱️  Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"[ChromaDB] 📈 Processing rate: {len(documents)/elapsed_time:.1f} documents/second")
    
    # 保存缓存
    _save_cache_file(persist_directory, collection_name, documents_hash)
    
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
