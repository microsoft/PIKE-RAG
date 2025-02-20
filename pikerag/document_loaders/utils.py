# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders.base import BaseLoader

from pikerag.document_loaders.common import DocumentType


def infer_file_type(file_path: str) -> Optional[DocumentType]:
    if os.path.exists(file_path):
        file_extension = Path(file_path).suffix[1:]
        for e in DocumentType:
            if file_extension in e.value:
                return e

        # TODO: move to logging instead
        print(f"File type cannot recognized: {file_path}.")
        print(f"Please check the pikerag.document_loaders.DocumentTyre for supported types.")
        return None

    else:
        # TODO: is it an url?
        pass

    return None


def get_loader(file_path: str, file_type: DocumentType = None) -> Optional[BaseLoader]:
    inferred_file_type = file_type
    if file_type is None:
        inferred_file_type = infer_file_type(file_path)
        if inferred_file_type is None:
            print(f"Cannot choose Document Loader with undefined type.")
            return None

    if inferred_file_type == DocumentType.csv:
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path, encoding="utf-8", autodetect_encoding=True)

    elif inferred_file_type == DocumentType.excel:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(file_path)

    elif inferred_file_type == DocumentType.markdown:
        from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
        return UnstructuredMarkdownLoader(file_path)

    elif inferred_file_type == DocumentType.text:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)

    elif inferred_file_type == DocumentType.word:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        return UnstructuredWordDocumentLoader(file_path)

    elif inferred_file_type == DocumentType.pdf:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        return UnstructuredPDFLoader(file_path)

    else:
        if file_type is not None:
            print(f"Document Loader for type {file_type} not defined.")
        else:
            print(f"Document Loader for inferred type {inferred_file_type} not defined.")
        return None
