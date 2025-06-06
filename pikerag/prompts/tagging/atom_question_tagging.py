# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate


DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant good at content understanding and asking question."
DEFAULT_SYSTEM_PROMPT_CN = "你是一个擅长内容理解和提问的有用AI助手。"


atom_question_tagging_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to extract as many questions as possible that are relevant and can be answered by the given content. Please try to be diverse and avoid extracting duplicated or similar questions. Make sure your question contain necessary entity names and avoid to use pronouns like it, he, she, they, the company, the person etc.

# Output Format
Output your answers line by line, with each question on a new line, without itemized symbols or numbers.

# Content
{content}

# Output:
""".strip()),
    ],
    input_variables=["content"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


atom_question_tagging_template_cn = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# 任务
您的任务是提取尽可能多的与给定内容相关且可以被该内容回答的问题。请尽量保持多样性，避免提取重复或相似的问题。确保您的问题包含必要的实体名称，避免使用代词如它、他、她、他们、该公司、该人等。

# 输出格式
逐行输出您的答案，每个问题占一行，不使用项目符号或数字。

# 内容
{content}

# 输出：
""".strip()),
    ],
    input_variables=["content"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT_CN,
    },
)


class AtomQuestionParser(BaseContentParser):
    def encode(self, content: str, **kwargs) -> Tuple[str, dict]:
        title = kwargs.get("title", None)
        if title is not None:
            content = f"Title: {title}. Content: {content}"
        return content, {}

    def decode(self, content: str, **kwargs) -> List[str]:
        questions = content.split("\n")
        questions = [question.strip() for question in questions if len(question.strip()) > 0]
        return questions


atom_question_tagging_protocol = CommunicationProtocol(
    template=atom_question_tagging_template,
    parser=AtomQuestionParser(),
)


atom_question_tagging_protocol_cn = CommunicationProtocol(
    template=atom_question_tagging_template_cn,
    parser=AtomQuestionParser(),
)
