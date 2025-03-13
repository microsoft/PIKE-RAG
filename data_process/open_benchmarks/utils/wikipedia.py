# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import requests
from typing import Any, Callable, Dict, List, Tuple
import urllib.parse

import aiohttp
import wikipediaapi
from wikipediaapi import WikipediaPage, WikipediaPageSection

from open_benchmarks.utils.io import dump_bytes_to_file, dump_texts_to_file, async_dump_bytes_to_file


WIKI_WIKI = wikipediaapi.Wikipedia('Microsoft Research Asia PIKE-RAG', 'en')


def get_raw_bytes(url: str) -> bytes:
    with requests.get(url) as response:
        assert response.status_code == 200, (
            "Url must be accessible since the given page is checked to be valid.\n"
            f"Response {response.status_code} to url: {url}"
        )
        ret = response.content
    return ret


def get_html_bytes(page: WikipediaPage) -> bytes:
    url = page.fullurl
    return get_raw_bytes(url)


def get_pdf_bytes(page: WikipediaPage) -> bytes:
    parsed_title = urllib.parse.quote(page.title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{parsed_title}"
    return get_raw_bytes(url)


def _extract_markdown_texts(sections: List[WikipediaPageSection], level: int) -> str:
    texts = ""
    for section in sections:
        title_prefix = "#" * level
        texts += f"{title_prefix} **{section.title}**\n\n"
        texts += f"{section.text}\n\n"
        texts += _extract_markdown_texts(section.sections, level + 1)
    return texts


def get_markdown_texts(page: WikipediaPage) -> str:
    texts = f"# **{page.title}**\n\n"
    texts += f"{page.summary.strip()}\n\n"
    texts += _extract_markdown_texts(page.sections, level=2)
    return texts


FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[WikipediaPage], Any]] = {
    "html": get_html_bytes,
    "pdf": get_pdf_bytes,
    "md": get_markdown_texts,
}


FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": dump_bytes_to_file,
    "pdf": dump_bytes_to_file,
    "md": dump_texts_to_file,
}


def download_all_titles(titles: List[str], dump_path_by_type_list: List[Dict[str, str]]) -> Tuple[bool, Dict[str, bool]]:
    """Try to download all wikipedia pages with the given titles. Download all or nothing, no partial downloads.

    Args:
        titles (List[str]):
        dump_path_by_type_list (List[Dict[str, str]]):

    Returns:
        bool: True if all wikipedia pages can be found and downloaded successfully, else False.
        Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is whether the
            page exists or not.
    """
    pages: List[WikipediaPage] = []
    title_valid: Dict[str, bool] = {}
    for title in titles:
        page = WIKI_WIKI.page(title)
        if not page.exists():
            title_valid[title] = False
            return False, title_valid
        pages.append(page)
        title_valid[title] = True

    for page, dump_path_by_type in zip(pages, dump_path_by_type_list):
        for filetype, dump_path in dump_path_by_type.items():
            obj = FILE_TYPE_TO_GET_FUNCTION[filetype](page)
            FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)

    return True, title_valid

################################################################################
## Async Implementation Below
################################################################################

async def async_get_raw_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


async def async_get_html_bytes(page: WikipediaPage) -> bytes:
    url = page.fullurl
    return await async_get_raw_bytes(url)


async def async_get_pdf_bytes(page: WikipediaPage) -> bytes:
    parsed_title = urllib.parse.quote(page.title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/pdf/{parsed_title}"
    return await async_get_raw_bytes(url)


ASYNC_FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[WikipediaPage], Any]] = {
    "html": async_get_html_bytes,
    "pdf": async_get_pdf_bytes,
}


ASYNC_FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": async_dump_bytes_to_file,
    "pdf": async_dump_bytes_to_file,
}


async def async_connect_and_save(page: WikipediaPage, filetype: str, dump_path: str):
    obj = await ASYNC_FILE_TYPE_TO_GET_FUNCTION[filetype](page)
    await ASYNC_FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)


async def async_download_all_titles(
    batch_titles: List[List[str]],
    batch_path_lists: List[List[Dict[str, str]]],
) -> List[Tuple[bool, Dict[str, bool]]]:
    """For each (titles, path_lists) pair in the given batch list, try to download all or nothing, no partial downloads.

    Args:
        batch_titles (List[List[str]]):
        batch_path_lists (List[List[Dict[str, str]]]):

    Returns:
        List[Tuple[bool, Dict[str, bool]]]: Each element is this list contains two items:
            - bool: True if all wikipedia pages can be found and downloaded successfully, else False.
            - Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is
                whether the page exists or not.
    """
    # I tried several methods here, async cannot be applied to check page exists, I only use it in downloading
    valid_batch_titles = []
    valid_batch_lists = []
    title_valid: Dict[str, bool] = {}

    # keep the batch that all pages exist
    for titles, batch_path in zip(batch_titles, batch_path_lists):
        list_all_exist = True
        valid_pages = []
        for title in titles:
            page = WIKI_WIKI.page(title)
            if not page.exists():
                list_all_exist = False
                title_valid[title] = False
                return False, title_valid
            valid_pages.append(page)
            title_valid[title] = True

        if list_all_exist:
            valid_batch_titles.extend(valid_pages)
            valid_batch_lists.extend(batch_path)

    # create download tasks
    download_tasks = []

    for page, dump_path_by_type in zip(valid_batch_titles, valid_batch_lists):
        for filetype, dump_path in dump_path_by_type.items():
            download_tasks.append(
                async_connect_and_save(page, filetype, dump_path)
            )

    await asyncio.gather(*download_tasks)

    return True, title_valid
