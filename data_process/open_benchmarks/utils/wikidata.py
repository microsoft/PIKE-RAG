# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import requests
from requests import Response
from typing import Any, Callable, Dict, List, Tuple

import aiohttp
from bs4 import BeautifulSoup

from open_benchmarks.utils.io import dump_bytes_to_file, dump_texts_to_file, async_dump_bytes_to_file


def parse_contents(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')

    title: str = soup.find("span", {"class": "wikibase-title-label"}).get_text()
    description: List[str] = []
    heading_desc: str = soup.find('span', class_='wikibase-descriptionview-text').get_text()
    description.append(heading_desc)
    extra_descriptions = soup.find_all('li', class_='wikibase-entitytermsview-aliases-alias')
    for desc in extra_descriptions:
        description.append(desc.get_text())

    statements: dict[str: List[str]] = {}
    statement_groups = soup.find_all(class_='wikibase-statementgroupview')
    for group in statement_groups:
        property_label = group.find(class_='wikibase-statementgroupview-property-label')
        if property_label:
            property_text = property_label.get_text().strip()
            values = []
            value_elements = group.find_all(class_='wikibase-snakview-value')
            # filter references here, I think its helpless
            for value in value_elements:
                if not value.find_parent(class_='wikibase-statementview-references-container'):
                    values.append(value.get_text().strip())
            statements[property_text] = values

    return {
        'title': title,
        'description': description,
        'statements': statements
    }


def contents_to_markdown_string(contents: dict) -> str:
    markdown_content = f"# {contents['title']}\n\n"

    for desc_idx in range(len(contents['description'])):
        if desc_idx != len(contents['description']) - 1:
            markdown_content += f"{contents['description'][desc_idx]} | "
        else:
            markdown_content += f"{contents['description'][desc_idx]}\n\n"

    markdown_content += '## **Statements**\n\n'
    for key, values in contents['statements'].items():
        markdown_content += f'### {key}:\n'
        for value in values:
            markdown_content += f'- {value}\n'

    return markdown_content


def get_html_bytes(response: Response) -> bytes:
    return response.content


def get_pdf_bytes(response: Response) -> bytes:
    # The `url` ends with ".json".
    qid = response.url.split("/")[-1].replace(".json", "")
    url = f"https://www.wikidata.org/api/rest_v1/page/pdf/{qid}"
    with requests.get(url) as response:
        assert response.status_code == 200, "Url must be accessible since the given qid is checked to be valid."
        ret = response.content
    return ret


def get_markdown_texts(response: Response) -> str:
    contents = parse_contents(response.text)
    texts = contents_to_markdown_string(contents)
    return texts


FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[Response], Any]] = {
    "html": get_html_bytes,
    "pdf": get_pdf_bytes,
    "md": get_markdown_texts,
}


FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": dump_bytes_to_file,
    "pdf": dump_bytes_to_file,
    "md": dump_texts_to_file,
}


def download_all_titles(
    titles: List[str], dump_path_by_type_list: List[Dict[str, str]], title2qid: Dict[str, str],
) -> Tuple[bool, Dict[str, bool]]:
    """Try to download all wikidata pages with the given titles. Download all or nothing, no partial downloads.

    Args:
        titles (List[str]):
        dump_path_by_type_list (List[Dict[str, str]]):

    Returns:
        bool: True if all wikidata pages can be found and downloaded successfully, else False.
        Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is whether the
            page exists or not.
    """
    responses: List[Response] = []
    title_valid: Dict[str, bool] = {}
    for title in titles:
        qid = title2qid.get(title, None)
        if qid is None:
            title_valid[title] = False
            return False, title_valid

        url = f"https://www.wikidata.org/wiki/{qid}"
        response = requests.get(url)
        if response.status_code != 200:
            title_valid[title] = False
            return False, title_valid
        responses.append(response)
        title_valid[title] = True

    for response, dump_path_by_type in zip(responses, dump_path_by_type_list):
        for filetype, dump_path in dump_path_by_type.items():
            obj = FILE_TYPE_TO_GET_FUNCTION[filetype](response)
            FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path)

    return True, title_valid

################################################################################
## Async Implementation Below
################################################################################

async def async_get_html_bytes(response: aiohttp.ClientResponse) -> bytes:
    return await response.read()


async def async_get_pdf_bytes(session: aiohttp.ClientSession, qid: str) -> bytes:
    url = f"https://www.wikidata.org/api/rest_v1/page/pdf/{qid}"
    async with session.get(url) as response:
        assert response.status == 200, "Url must be accessible since the given qid is checked to be valid."
        return await response.read()


ASYNC_FILE_TYPE_TO_GET_FUNCTION: Dict[str, Callable[[aiohttp.ClientResponse, aiohttp.ClientSession, str], Any]] = {
    "html": async_get_html_bytes,
    "pdf": async_get_pdf_bytes
}


ASYNC_FILE_TYPE_TO_DUMP_FUNCTION: Dict[str, Callable[[Any, str], None]] = {
    "html": async_dump_bytes_to_file,
    "pdf": async_dump_bytes_to_file
}


async def async_fetch_response(session: aiohttp.ClientSession, title: str, title2qid: Dict[str, str]) -> Tuple[str, aiohttp.ClientResponse, bool]:
    qid = title2qid.get(title, None)
    if qid is None:
        return title, None, False

    url = f"https://www.wikidata.org/wiki/{qid}"
    async with session.get(url) as response:
        if response.status != 200:
            return title, None, False
        return title, response, True


async def async_download_all_titles(
    titles: List[str], dump_path_by_type_list: List[Dict[str, str]], title2qid: Dict[str, str],
) -> Tuple[bool, Dict[str, bool]]:
    """Try to download all Wikidata pages with the given titles. Download all or nothing, no partial downloads.

    Args:
        titles (List[str]):
        dump_path_by_type_list (List[Dict[str, str]]):

    Returns:
        bool: True if all Wikidata pages can be found and downloaded successfully, else False.
        Dict[str, bool]: The key is the tile str that has been accessed in this function call, the value is whether the
            page exists or not.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [async_fetch_response(session, title, title2qid) for title in titles]
        results = await asyncio.gather(*tasks)

    responses: List[Tuple[str, aiohttp.ClientResponse]] = []
    title_valid: Dict[str, bool] = {}

    for title, response, is_valid in results:
        title_valid[title] = is_valid
        if not is_valid:
            return False, title_valid
        responses.append((title, response))

    download_tasks = []

    async with aiohttp.ClientSession() as session:
        for (title, response), dump_path_by_type in zip(responses, dump_path_by_type_list):
            qid = title2qid[title]
            for filetype, dump_path in dump_path_by_type.items():
                if filetype == "html":
                    obj = await ASYNC_FILE_TYPE_TO_GET_FUNCTION[filetype](response)
                elif filetype == "pdf":
                    obj = await ASYNC_FILE_TYPE_TO_GET_FUNCTION[filetype](session, qid)
                download_tasks.append(ASYNC_FILE_TYPE_TO_DUMP_FUNCTION[filetype](obj, dump_path))

        await asyncio.gather(*download_tasks)

    return True, title_valid
