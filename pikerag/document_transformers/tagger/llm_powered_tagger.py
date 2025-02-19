# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Sequence

from tqdm import tqdm

from langchain_core.documents import Document, BaseDocumentTransformer

from pikerag.llm_client import BaseLLMClient
from pikerag.prompts import CommunicationProtocol
from pikerag.utils.logger import Logger


class LLMPoweredTagger(BaseDocumentTransformer):
    NAME = "LLMPoweredTagger"

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tagging_protocol: CommunicationProtocol,
        num_parallel: int=1,
        tag_name: str = "tags",
        llm_config: dict = {},
        logger: Logger = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self._llm_client = llm_client
        self._llm_config = llm_config

        self._num_parallel: int = num_parallel

        self._tagging_protocol = tagging_protocol
        self._tag_name = tag_name

        self.logger = logger

    def _get_tags_info(self, content: str, **metadata) -> List[Any]:
        messages = self._tagging_protocol.process_input(content=content, **metadata)

        # Call client for tags
        response = self._llm_client.generate_content_with_messages(messages=messages, **self._llm_config)

        return self._tagging_protocol.parse_output(content=response, **metadata)

    def _single_thread_transform(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        ret_docs: List[Document] = []
        for idx, doc in tqdm(enumerate(documents), desc="Tagging Documents", total=len(documents)):
            content = doc.page_content
            metadata = doc.metadata

            tags = self._get_tags_info(content, **metadata)
            if self.logger is not None:
                self.logger.debug(f"{idx + 1}/{len(documents)} document -- tags: {tags}", tag=self.NAME)

            full_tags = metadata.get(self._tag_name, []) + tags
            metadata.update({self._tag_name: full_tags})
            ret_docs.append(doc)
        return ret_docs

    def _multiple_threads_transform(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        self.logger.info(f"Tagging {len(documents)} with parallel level set to {self._num_parallel}.")

        pbar = tqdm(total=len(documents), desc="Tagging Documents")
        # Create a ThreadPoolExecutor to manage a pool of threads
        with ThreadPoolExecutor(max_workers=self._num_parallel) as executor:
            # Submit all documents to the executor
            future_to_index = {
                executor.submit(self._get_tags_info, doc.page_content, **doc.metadata): idx
                for idx, doc in enumerate(documents)
            }

            # Process futures as they complete
            ret_docs: List[Document] = [None] * len(documents)
            for future in as_completed(future_to_index):
                doc_idx = future_to_index[future]
                doc = documents[doc_idx]
                try:
                    tags = future.result()
                    full_tags = doc.metadata.get(self._tag_name, []) + tags
                    doc.metadata.update({self._tag_name: full_tags})
                except Exception as e:
                    pass

                ret_docs[doc_idx] = doc

                pbar.update(1)

        pbar.close()

        return ret_docs

    # TODO: create new interface like "TextSplitter" for tagging?
    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        if self._num_parallel == 1:
            return self._single_thread_transform(documents, **kwargs)
        else:
            return self._multiple_threads_transform(documents, **kwargs)
