# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate


DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant good at content understanding and asking question."


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
