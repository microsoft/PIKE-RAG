# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pikerag.prompts.base_parser import BaseContentParser
from pikerag.prompts.chunking.resplit_parser import ResplitParser
from pikerag.prompts.message_template import MessageTemplate
from pikerag.prompts.protocol import CommunicationProtocol


chunk_summary_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at document summarization."),
        ("user", """
# Source of the original text

The original text comes from {filename}。

# Original text

"partial original text":
{content}

# Task

Your task is to summarize the above "partial original text"

# Output

The output should contain the summary, do not add any redundant information.
""".strip()),
    ],
    input_variables=["filename", "content"],
)


chunk_summary_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长文档摘要的有用AI助手。"),
        ("user", """
# 原文来源

原文来自{filename}。

# 原文

"部分原文"：
{content}

# 任务

您的任务是总结上述"部分原文"

# 输出

输出应包含摘要，不要添加任何冗余信息。
""".strip()),
    ],
    input_variables=["filename", "content"],
)


chunk_summary_refinement_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at summary refinement."),
        ("user", """
# Source of the original text

The original text comes from {filename}。

# Original text

generalization of "partial original text":
{summary}

"partial original text":
{content}

# Task

Your task is to summarize the above "partial original text"

# Output

The output should contain the summary, do not add any redundant information.
""".strip()),
    ],
    input_variables=["filename", "summary", "content"],
)


chunk_summary_refinement_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长摘要精炼的有用AI助手。"),
        ("user", """
# 原文来源

原文来自{filename}。

# Original text

"部分原文"的概括：
{summary}

"部分原文"：
{content}

# Task

Your task is to summarize the above "partial original text"

# Output

The output should contain the summary, do not add any redundant information.
""".strip()),
    ],
    input_variables=["filename", "summary", "content"],
)


chunk_resplit_template = MessageTemplate(
    template=[
        ("system", "You are a helpful AI assistant good at document chunking."),
        ("user", """
# Source of the original text

The original text comes from {filename}。

# Original text

generalization of "the first part" of "partial original text":
{summary}

"partial original text":
{content}

# Task

Your task:
1. Understand the generalization of "the first part" of "partial original text" and the "partial original text";
2. Analyse the structure of "partial original text", Split the "partial original text" strictly into "the first part" and "the second part", no content can be missing.
3. Provide the "end line number" of "the first part", pay attention that "the first part" is defined as: all the content of "partial original text" from Line "0" to Line "end line number" + 1, where empty is not allowed. Please note that here the maximum line number is {max_line_number}.
4. Summarize "the first part"。
5. For "the second part", considering the context and summarizing the main content of "the first part", please note that the content of "the first part" is defined as: all "partial original text" content after Line "end line number" + 1.

# Output

The output should strictly follow the format below, do not add any redundant information.

Thinking: According to the task requirements, carefully analyze the structure of the above "partial original text", think about how to reasonably split it into two parts, and output your thinking process.

<result>
<chunk>
  <endline>end line number, a non-negative number indicates the end line of "the first part". The first part will include this line.</endline>
  <summary>A summary of the "first part". Starting with "The main content of this part is". It can be referred to the generalization of "partial original text"</summary>
</chunk>
<chunk>
  <summary>Combine the context and the generalization of "the first part" to summarize the main content of "the second part". Starting with "The main content of this part is".</summary>
</chunk>
</result>
""".strip()),
    ],
    input_variables=["filename", "summary", "content", "max_line_number"],
)


chunk_resplit_template_cn = MessageTemplate(
    template=[
        ("system", "你是一个擅长文档分块的有用AI助手。"),
        ("user", """
# 原文来源

原文来自{filename}。

# 原文

"部分原文"的"第一部分"的概括：
{summary}

"部分原文"：
{content}

# 任务

您的任务：
1. 理解"部分原文"的"第一部分"的概括和"部分原文"；
2. 分析"部分原文"的结构，将"部分原文"严格分割成"第一部分"和"第二部分"，不能遗漏任何内容。
3. 提供"第一部分"的"结束行号"，注意"第一部分"的定义是：从第"0"行到第"结束行号"+1行的"部分原文"的所有内容，其中不允许为空。请注意这里的最大行号是{max_line_number}。
4. 总结"第一部分"。
5. 对于"第二部分"，考虑上下文和总结"第一部分"的主要内容，请注意"第一部分"的内容定义为：第"结束行号"+1行之后的所有"部分原文"内容。

# 输出

输出应严格遵循以下格式，不要添加任何冗余信息。

思考：根据任务要求，仔细分析上述"部分原文"的结构，思考如何合理地将其分成两部分，并输出您的思考过程。

<result>
<chunk>
  <endline>结束行号，一个非负数，表示"第一部分"的结束行。第一部分将包含此行。</endline>
  <summary>"第一部分"的摘要。以"这部分的主要内容是"开头。可以参考"部分原文"的概括</summary>
</chunk>
<chunk>
  <summary>结合上下文和"第一部分"的概括来总结"第二部分"的主要内容。以"这部分的主要内容是"开头。</summary>
</chunk>
</result>
""".strip()),
    ],
    input_variables=["filename", "summary", "content", "max_line_number"],
)


chunk_summary_protocol = CommunicationProtocol(
    template=chunk_summary_template,
    parser=BaseContentParser(),
)


chunk_summary_protocol_cn = CommunicationProtocol(
    template=chunk_summary_template_cn,
    parser=BaseContentParser(),
)


chunk_summary_refinement_protocol = CommunicationProtocol(
    template=chunk_summary_refinement_template,
    parser=BaseContentParser(),
)


chunk_summary_refinement_protocol_cn = CommunicationProtocol(
    template=chunk_summary_refinement_template_cn,
    parser=BaseContentParser(),
)


chunk_resplit_protocol = CommunicationProtocol(
    template=chunk_resplit_template,
    parser=ResplitParser(),
)


chunk_resplit_protocol_cn = CommunicationProtocol(
    template=chunk_resplit_template_cn,
    parser=ResplitParser(),
)
