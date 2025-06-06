# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.lxml_parser import get_soup_from_content


semantic_tagging_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} that can help people {task_direction}."),
        ("user", """
# Task
Please read the content provided carefully, think step by step, then extract the {tag_semantic} phrases contained therein.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given content.</thinking>
  <phrases>
    <phrase>Extracted phrase 1</phrase>
    <phrase>Extracted phrase 2</phrase>
    <phrase>Extracted phrase 3</phrase>
    ... Please output an equal number of phrases based on the number of phrases contained in the content. Leave it empty if no phrase found.
  </phrases>
</result>

# Content
{content}

# Thinking and answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "tag_semantic", "content"],
)


semantic_tagging_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个在{knowledge_domain}领域很有帮助的助手，能够帮助人们{task_direction}。"),
        ("user", """
# 任务
请仔细阅读提供的内容，逐步思考，然后提取其中包含的{tag_semantic}短语。

# 输出格式
输出应严格遵循以下格式，不要添加任何冗余信息。

<result>
  <thinking>您对给定内容的思考过程。</thinking>
  <phrases>
    <phrase>提取的短语1</phrase>
    <phrase>提取的短语2</phrase>
    <phrase>提取的短语3</phrase>
    ... 请根据内容中包含的短语数量输出相等数量的短语。如果没有找到短语，请留空。
  </phrases>
</result>

# 内容
{content}

# 思考和回答
""".strip()),
    ],
    input_variables=["knowledge_domain", "task_direction", "tag_semantic", "content"],
)


class SemanticTaggingParser(BaseContentParser):
    def decode(self, content: str, **kwargs) -> List[str]:
        thinking: str = ""
        phrases: List[str] = []

        result_soup: BeautifulSoup = get_soup_from_content(content=content, tag="result")
        if result_soup is not None:
            thinking_soup = result_soup.find("thinking")
            phrases_soup = result_soup.find("phrases")

            if thinking_soup is not None:
                thinking = thinking_soup.text.strip()

            if phrases_soup is not None:
                for phrase_soup in phrases_soup.find_all("phrase"):
                    phrase_str = phrase_soup.text.strip()
                    if len(phrase_str) > 0:
                        phrases.append(phrase_str)

            else:
                # TODO: add logger for Parser?
                print(f"[SemanticTagParser] Content skipped due to the absence of <phrases>: {content}")

        else:
            # TODO: add logger for Parser?
            print(f"[SemanticTagParser] Content skipped due to the absence of <result>: {content}")

        # NOTE: thinking not returned to let the return value compatible with LLMPoweredTagger.
        return phrases


semantic_tagging_protocol = CommunicationProtocol(
    template=semantic_tagging_template,
    parser=SemanticTaggingParser(),
)


semantic_tagging_protocol_cn = CommunicationProtocol(
    template=semantic_tagging_template_cn,
    parser=SemanticTaggingParser(),
)
