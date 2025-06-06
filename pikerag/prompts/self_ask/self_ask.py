# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import Dict, List, Optional, Tuple

from pikerag.prompts import BaseContentParser, CommunicationProtocol, MessageTemplate


IntermediateStop: str = "Intermediate answer:"


self_ask_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at question-answering."),
        ("user", """
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Are follow up questions needed here: Yes.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
Are follow up questions needed here: No.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Are follow up questions needed here: Yes.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
Are follow up questions needed here: No.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Are follow up questions needed here: Yes.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
Are follow up questions needed here: No.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Are follow up questions needed here: Yes.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Are follow up questions needed here: Yes.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Are follow up questions needed here: Yes.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
Are follow up questions needed here: No.
So the final answer is: No

Question: {content}
{followup_context}
{asking_prefix}
""".strip()),
    ],
    input_variables=["content", "followup_context", "asking_prefix"],
)


self_ask_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长问答的有用AI助手。"),
        ("user", """
问题：谁活得更久，穆罕默德·阿里还是艾伦·图灵？
这里需要后续问题吗：是的。
后续问题：穆罕默德·阿里去世时多大年纪？
中间答案：穆罕默德·阿里去世时74岁。
这里需要后续问题吗：是的。
后续问题：艾伦·图灵去世时多大年纪？
中间答案：艾伦·图灵去世时41岁。
这里需要后续问题吗：不需要。
所以最终答案是：穆罕默德·阿里

问题：craigslist的创始人是什么时候出生的？
这里需要后续问题吗：是的。
后续问题：谁是craigslist的创始人？
中间答案：Craigslist是由Craig Newmark创立的。
这里需要后续问题吗：是的。
后续问题：Craig Newmark是什么时候出生的？
中间答案：Craig Newmark出生于1952年12月6日。
这里需要后续问题吗：不需要。
所以最终答案是：1952年12月6日

问题：乔治·华盛顿的外祖父是谁？
这里需要后续问题吗：是的。
后续问题：乔治·华盛顿的母亲是谁？
中间答案：乔治·华盛顿的母亲是Mary Ball Washington。
这里需要后续问题吗：是的。
后续问题：Mary Ball Washington的父亲是谁？
中间答案：Mary Ball Washington的父亲是Joseph Ball。
这里需要后续问题吗：不需要。
所以最终答案是：Joseph Ball

问题：《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？
这里需要后续问题吗：是的。
后续问题：《大白鲨》的导演是谁？
中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
这里需要后续问题吗：是的。
后续问题：史蒂文·斯皮尔伯格来自哪里？
中间答案：美国。
这里需要后续问题吗：是的。
后续问题：《皇家赌场》的导演是谁？
中间答案：《皇家赌场》的导演是Martin Campbell。
这里需要后续问题吗：是的。
后续问题：Martin Campbell来自哪里？
中间答案：新西兰。
这里需要后续问题吗：不需要。
所以最终答案是：不是

问题：{content}
{followup_context}
{asking_prefix}
""".strip()),
    ],
    input_variables=["content", "followup_context", "asking_prefix"],
)


class SelfAskParser(BaseContentParser):
    def __init__(self) -> None:
        self._final_answer_prefix = "Are follow up questions needed here: No.\nSo the final answer is: "
        self._final_answer_prefix_cn = "这里需要后续问题吗：不需要。\n所以最终答案是："
        self._final_answer_pattern = re.compile(r"So the final answer is:(.*)", re.DOTALL)
        self._final_answer_pattern_cn = re.compile(r"所以最终答案是：(.*)", re.DOTALL)
        self._follow_up_prefix = "Are follow up questions needed here: "
        self._follow_up_prefix_cn = "这里需要后续问题吗："
        self._follow_up_pattern = re.compile(r"Follow up:(.*)", re.DOTALL)
        self._follow_up_pattern_cn = re.compile(r"后续问题：(.*)", re.DOTALL)

        self._ask_final: bool = False
        self._use_chinese: bool = False

    def encode(
        self, content: str, followup_pairs: List[Tuple[str, str]], ask_followup: bool, ask_final: bool, use_chinese: bool = False, **kwargs,
    ) -> Tuple[str, Dict]:
        self._use_chinese = use_chinese
        
        if use_chinese:
            followup_context: str = "\n".join(
                [
                    f"这里需要后续问题吗：是的。\n后续问题：{q}\n中间答案：{a}"
                    for q, a in followup_pairs
                ]
            )
        else:
            followup_context: str = "\n".join(
                [
                    f"Are follow up questions needed here: Yes.\nFollow up: {q}\nIntermediate Answer: {a}"
                    for q, a in followup_pairs
                ]
            )

        assert ask_followup != ask_final, f"There should be and should only be one `True` for `ask_followup` and `ask_final`"

        if len(followup_pairs) >= 5 and ask_followup is True:
            ask_followup = False
            ask_final = True

        self._ask_final = ask_final
        if ask_followup is True:
            asking_prefix = self._follow_up_prefix_cn if use_chinese else self._follow_up_prefix
        else:
            asking_prefix = self._final_answer_prefix_cn if use_chinese else self._final_answer_prefix

        return content, {
            "followup_context": followup_context,
            "asking_prefix": asking_prefix,
        }

    def decode(self, content: str,  **kwargs) -> Tuple[Optional[str], Optional[str]]:
        if isinstance(content, str):
            content = content.strip()

            if self._ask_final is False:
                # Try Chinese patterns first if using Chinese
                if self._use_chinese:
                    follow_up_match = re.search(self._follow_up_pattern_cn, content)
                    if follow_up_match is not None:
                        follow_up = follow_up_match.group(1)
                        return None, follow_up

                    final_answer_match = re.search(self._final_answer_pattern_cn, content)
                    if final_answer_match is not None:
                        final_answer = final_answer_match.group(1)
                        return final_answer, None

                # Fallback to English patterns
                follow_up_match = re.search(self._follow_up_pattern, content)
                if follow_up_match is not None:
                    follow_up = follow_up_match.group(1)
                    return None, follow_up

                final_answer_match = re.search(self._final_answer_pattern, content)
                if final_answer_match is not None:
                    final_answer = final_answer_match.group(1)
                    return final_answer, None

            else:
                return content, None

        return None, None


self_ask_protocol = CommunicationProtocol(
    template=self_ask_template,
    parser=SelfAskParser(),
)


self_ask_protocol_cn = CommunicationProtocol(
    template=self_ask_template_cn,
    parser=SelfAskParser(),
)
