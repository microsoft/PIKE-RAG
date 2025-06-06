# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import traceback
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate
from pikerag.utils.lxml_parser import get_soup_from_content


multiple_choice_qa_template = MessageTemplate(
    template=[
        ("system", "You are a helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option from the given options, the chosen option should be correct and the most suitable one to answer the given question. If you don't have sufficient data to determine, randomly choose one option from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask. Please note that only one single mask is allowable.</mask>
    <option>The option detail corresponds to the chosen option mask.</option>
  </answer>
</result>

# Question
{content}

# Options
{options_str}

# Thinking and Answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str"],
)


multiple_choice_qa_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长{knowledge_domain}知识的有用助手，能够帮助人们回答{knowledge_domain}问题。"),
        ("user", """
# 任务
您的任务是逐步思考，然后从给定选项中选择正确的选项，所选选项应该是正确的且最适合回答给定问题的。如果您没有足够的数据来确定，请从给定选项中随机选择一个选项。

# 输出格式
输出应严格遵循以下格式，不要添加任何冗余信息。

<result>
  <thinking>您对给定问题的思考过程。</thinking>
  <answer>
    <mask>选择的选项掩码。请注意只允许一个掩码。</mask>
    <option>与所选选项掩码对应的选项详情。</option>
  </answer>
</result>

# 问题
{content}

# 选项
{options_str}

# 思考和回答
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str"],
)


multiple_choice_qa_with_reference_template = MessageTemplate(
    template=[
        ("system", "You are an helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option from the given options, the chosen option should be correct and the most suitable one to answer the given question. You can refer to the references provided when thinking and answering. Please note that the references may or may not be relevant to the question. If you don't have sufficient information to determine, randomly choose one option from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask. Please note that only one single mask is allowable.</mask>
    <option>The option detail corresponds to the chosen option mask.</option>
  </answer>
</result>

# Question
{content}

# Options
{options_str}

# References
{references_str}

# Thinking and Answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str", "references_str"],
)


multiple_choice_qa_with_reference_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长{knowledge_domain}知识的有用助手，能够帮助人们回答{knowledge_domain}问题。"),
        ("user", """
# 任务
您的任务是逐步思考，然后从给定选项中选择正确的选项，所选选项应该是正确的且最适合回答给定问题的。
         在思考和回答时可以参考提供的参考资料。请注意，参考资料可能与问题相关，也可能不相关。
         如果您没有足够的信息来确定，请从给定选项中随机选择一个选项。

# 输出格式
输出应严格遵循以下格式，不要添加任何冗余信息。

<result>
  <thinking>您对给定问题的思考过程。</thinking>
  <answer>
    <mask>选择的选项掩码。请注意只允许一个掩码。</mask>
    <option>与所选选项掩码对应的选项详情。</option>
  </answer>
</result>

# 问题
{content}

# 选项
{options_str}

# 参考资料
{references_str}

# 思考和回答
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str", "references_str"],
)


# TODO: update the template & protocol to fit for both single choice and multiple choice.
multiple_choice_qa_with_reference_and_review_template = MessageTemplate(
    template=[
        ("system", "You are an helpful assistant good at {knowledge_domain} knowledge that can help people answer {knowledge_domain} questions."),
        ("user", """
# Task
Your task is to think step by step and then choose the correct option from the given options, the chosen option should be correct and the most suitable one to answer the given question. You can refer to the references provided when thinking and answering. Please note that the references may or may not be relevant to the question. If you don't have sufficient information to determine, randomly choose one option from the given options.

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask. Please note that only one single mask is allowable.</mask>
    <option>The option detail corresponds to the chosen option mask.</option>
  </answer>
</result>

# Question
{content}

# Options
{options_str}

# References
{references_str}

# Review
Let's now review the question, options and output format again:

# Question
{content}

# Options
{options_str}

# Output format
The output should strictly follow the format below, do not add any redundant information.

<result>
  <thinking>Your thinking for the given question.</thinking>
  <answer>
    <mask>The chosen option mask. Please note that only one single mask is allowable.</mask>
    <option>The option detail corresponds to the chosen option mask.</option>
  </answer>
</result>

# Thinking and Answer
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str", "references_str"],
)


# TODO: update the template & protocol to fit for both single choice and multiple choice.
multiple_choice_qa_with_reference_and_review_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长{knowledge_domain}知识的有用助手，能够帮助人们回答{knowledge_domain}问题。"),
        ("user", """
# 任务
您的任务是逐步思考，然后从给定选项中选择正确的选项，所选选项应该是正确的且最适合回答给定问题的。在思考和回答时可以参考提供的参考资料。请注意，参考资料可能与问题相关，也可能不相关。如果您没有足够的信息来确定，请从给定选项中随机选择一个选项。

# 输出格式
输出应严格遵循以下格式，不要添加任何冗余信息。

<result>
  <thinking>您对给定问题的思考过程。</thinking>
  <answer>
    <mask>选择的选项掩码。请注意只允许一个掩码。</mask>
    <option>与所选选项掩码对应的选项详情。</option>
  </answer>
</result>

# 问题
{content}

# 选项
{options_str}

# 参考资料
{references_str}

# 回顾
让我们现在再次回顾问题、选项和输出格式：

# 问题
{content}

# 选项
{options_str}

# 输出格式
输出应严格遵循以下格式，不要添加任何冗余信息。

<result>
  <thinking>您对给定问题的思考过程。</thinking>
  <answer>
    <mask>选择的选项掩码。请注意只允许一个掩码。</mask>
    <option>与所选选项掩码对应的选项详情。</option>
  </answer>
</result>

# 思考和回答
""".strip()),
    ],
    input_variables=["knowledge_domain", "content", "options_str", "references_str"],
)


class MultipleChoiceQaParser(BaseContentParser):
    def __init__(self) -> None:
        self.option_masks: List[str] = []
        self.options: Dict[str, str] = {}

    def encode(self, content: str, options: Dict[str, str], answer_mask_labels: List[str], **kwargs) -> Tuple[str, dict]:
        self.option_masks = sorted(list(options.keys()))
        self.options = options.copy()

        # NOTE: could enable re-ordering method in the future, do remember to check the answer mask as well.
        options_str = "\n".join([f"{key}: {self.options[key]}" for key in self.option_masks])

        for mask_label in answer_mask_labels:
            assert mask_label in self.option_masks, (
                f"Given answer mask label {mask_label}, but no corresponding option provided: {self.option_masks}"
            )

        return content, {"options_str": options_str}

    # TODO: update the decode interface to be Tuple[answer, dict]
    def decode(self, content: str, options: Dict[str, str], **kwargs) -> dict:
        if content is None or content == "":
            return {}

        try:
            result_soup: BeautifulSoup = get_soup_from_content(content, tag="result")
            if result_soup is not None:
                thinking_soup = result_soup.find("thinking")
                answer_soup = result_soup.find("answer")
            else:
                thinking_soup = get_soup_from_content(content, tag="thinking")
                answer_soup = get_soup_from_content(content, "answer")

            if thinking_soup is not None:
                thinking = thinking_soup.text
            else:
                thinking = ""

            if answer_soup is not None:
                mask_soup = answer_soup.find("mask")
                mask = mask_soup.text.strip() if mask_soup is not None else ""
                option_soup = answer_soup.find("option")
                option = option_soup.text.strip() if option_soup is not None else ""
            else:
                mask = ""
                option = ""

            if len(mask) == 1:
                assert mask in self.option_masks, f"choose {mask} from {self.option_masks}\n{content}"
                if option != self.options[mask]:
                    print()
                    print(f"Answer option: [{option}]")
                    print(f"But the Given: [{self.options[mask]}]")
            elif len(mask) == 0:
                print("No mask extracted")
            else:
                print(f"Multiple options chosen: {mask}")

        except Exception as e:
            print("Content:")
            print(content)
            print("Exception")
            print(e)
            traceback.print_exc()
            exit(0)

        return {
            "thinking": thinking,
            "answer": mask,
            "chosen_option": option,
        }


class MultipleChoiceQaWithReferenceParser(MultipleChoiceQaParser):
    def encode(self, content: str, options: Dict[str, str], answer_mask_labels: List[str], **kwargs) -> Tuple[str, Dict]:
        content, supplementary = super().encode(content, options, answer_mask_labels[0], **kwargs)

        references = kwargs.get("references", [])
        supplementary["references_str"] = "\n".join([reference.strip() for reference in references])

        return content, supplementary


multiple_choice_qa_protocol = CommunicationProtocol(
    template=multiple_choice_qa_template,
    parser=MultipleChoiceQaParser(),
)


multiple_choice_qa_protocol_cn = CommunicationProtocol(
    template=multiple_choice_qa_template_cn,
    parser=MultipleChoiceQaParser(),
)


multiple_choice_qa_with_reference_protocol = CommunicationProtocol(
    template=multiple_choice_qa_with_reference_template,
    parser=MultipleChoiceQaWithReferenceParser(),
)


multiple_choice_qa_with_reference_protocol_cn = CommunicationProtocol(
    template=multiple_choice_qa_with_reference_template_cn,
    parser=MultipleChoiceQaWithReferenceParser(),
)


multiple_choice_qa_with_reference_and_review_protocol = CommunicationProtocol(
    template=multiple_choice_qa_with_reference_and_review_template,
    parser=MultipleChoiceQaWithReferenceParser(),
)


multiple_choice_qa_with_reference_and_review_protocol_cn = CommunicationProtocol(
    template=multiple_choice_qa_with_reference_and_review_template_cn,
    parser=MultipleChoiceQaWithReferenceParser(),
)
