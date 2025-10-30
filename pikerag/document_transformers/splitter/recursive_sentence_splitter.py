# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import List, Optional

import spacy
import spacy.tokens
from tqdm import tqdm

from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document


LANG2MODELNAME = {
    "en": "en_core_web_lg",
    "zh": "zh_core_web_lg",
}


class RecursiveSentenceSplitter(TextSplitter):
    NAME = "RecursiveSentenceSplitter"

    def __init__(
        self,
        lang: str = "en",
        nlp_max_len: int = 4000000,
        num_parallel: int = 4,
        chunk_size: int = 12,
        chunk_overlap: int = 4,
        **kwargs,
    ):
        """
        Args:
            lang (str): "en" for English, "zh" for Chinese.
            nlp_max_len (int):
            num_workers (int):
            chunk_size (int): number of sentences per chunk.
            chunk_over_lap (int): number of sentence overlap between two continuous chunks.
        """
        super().__init__(chunk_size, chunk_overlap)
        self._stride: int = self._chunk_size - self._chunk_overlap
        self._num_parallel: int = num_parallel

        self._load_model(lang, nlp_max_len)

        return

    def _load_model(self, language: str, nlp_max_len: int) -> None:
        assert language in LANG2MODELNAME, f"Spacy model not specified for language: {language}."

        model_name = LANG2MODELNAME[language]
        try:
            self._nlp = spacy.load(model_name)
        except:
            spacy.cli.download(model_name)
            self._nlp = spacy.load(model_name)
        self._nlp.max_length = nlp_max_len
        return

    def _nlp_doc_to_texts(self, doc: spacy.tokens.Doc) -> List[str]:
        sents = [sent.text.strip() for sent in doc.sents]
        sents = [sent for sent in sents if len(sent) > 0]

        segments: List[str] = []
        for i in range(0, len(sents), self._stride):
            segment = " ".join(sents[i : i + self._chunk_size])
            segments.append(segment)
            if i + self._chunk_size >= len(sents):
                break

        return segments

    def split_text(self, text: str) -> List[str]:
        doc = self._nlp(text)
        segments = self._nlp_doc_to_texts(doc)
        return segments

    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        pbar = tqdm(total=len(texts), desc="Splitting texts by sentences")
        num_workers = min(len(texts), self._num_parallel)
        for idx, doc in enumerate(self._nlp.pipe(texts, n_process=num_workers, batch_size=32)):
            segments = self._nlp_doc_to_texts(doc)
            for segment in segments:
                documents.append(Document(page_content=segment, metadata=deepcopy(_metadatas[idx])))
            pbar.update(1)
        pbar.close()

        return documents
