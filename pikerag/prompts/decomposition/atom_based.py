# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo
from pikerag.prompts import MessageTemplate, BaseContentParser, CommunicationProtocol
from pikerag.prompts.qa.generation import generation_qa_with_reference_template, GenerationQaParser
from pikerag.utils.json_parser import parse_json


DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant on question answering."


def atom_infos_to_context_string(chosen_atom_infos: List[AtomRetrievalInfo], limit: int=80000) -> str:
    context: str = ""
    chunk_id_set = set()
    for info in chosen_atom_infos:
        if info.source_chunk_id in chunk_id_set:
            continue
        chunk_id_set.add(info.source_chunk_id)

        if info.source_chunk_title is not None:
            context += f"\nTitle: {info.source_chunk_title}. Content: {info.source_chunk}\n"
        else:
            context += f"\n{info.source_chunk}\n"

        if len(context) >= limit:
            break

    context = context.strip()
    return context

################################################################################

question_decomposition_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to analyse the providing context then raise atomic sub-questions for the knowledge that can help you answer the question better. Think in different ways and raise as many diverse questions as possible.

# Output Format
Please output in following JSON format:
{{
    "thinking": <A string. Your thinking for this task, including analysis to the question and the given context.>,
    "sub_questions": <A list of string. The sub-questions indicating what you need.>
}}

# Context
The context we already have:
{chosen_context}

# Question
{content}

# Your Output:
""".strip()),
    ],
    input_variables=["content", "chosen_context"],
    partial_variables={
        "system_prompt": "You are a helpful AI assistant good at question decomposition.",
    },
)


class QuestionDecompositionParser(BaseContentParser):
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, dict]:
        context = atom_infos_to_context_string(chosen_atom_infos)
        return content, {"chosen_context": context}

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, List[str]]:
        try:
            output = parse_json(content)

            thinking: str = output["thinking"]
            sub_questions = output["sub_questions"]
            return len(sub_questions) > 0, thinking, sub_questions
        except Exception as e:
            print(f"[QuestionDecompositionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", []


question_decompose_protocol = CommunicationProtocol(
    template=question_decomposition_template,
    parser=QuestionDecompositionParser(),
)

################################################################################

atom_question_selection_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to analyse the providing context then decide which sub-questions may be useful to be answered before you can answer the given question. Select a most relevant sub-question from the given question list, avoid selecting sub-question that can already be answered with the given context or with your own knowledge.

# Output Format
Please output in following JSON format:
{{
    "thinking": <A string. Your thinking for this selection task.>,
    "question_idx": <A list of integers. Each integer indicates a sub-question index from 1 to {num_atoms}.>
}}

# Context
The context we already have:
{chosen_context}

# Sub-Questions You Can Choose From
{atom_list_str}

# Question
{content}

# Your output:
""".strip()),
    ],
    input_variables=["content", "num_atoms", "chosen_context", "atom_list_str"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class AtomQuestionSelectionParser(BaseContentParser):
    def __init__(self) -> None:
        super().__init__()
        self._atom_info_candidates: List[AtomRetrievalInfo] = []

    def encode(
        self, content: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo], **kwargs,
    ) -> Tuple[str, dict]:
        context = atom_infos_to_context_string(chosen_atom_infos)

        atom_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            atom_list_str += f"Question {i + 1}: {info.atom}\n"

        self._atom_info_candidates = atom_info_candidates

        return content, {
            "num_atoms": len(atom_info_candidates),
            "chosen_context": context,
            "atom_list_str": atom_list_str,
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, AtomRetrievalInfo]:
        try:
            output = parse_json(content)
            thinking: str = output["thinking"]
            question_idx = output["question_idx"]
            if question_idx is not None and question_idx > 0 and question_idx <= len(self._atom_info_candidates):
                chosen_info = self._atom_info_candidates[question_idx - 1]
                return True, thinking, chosen_info
            else:
                return False, thinking, None
        except Exception as e:
            print(f"[AtomQuestionSelectionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", None


atom_question_selection_protocol = CommunicationProtocol(
    template=atom_question_selection_template,
    parser=AtomQuestionSelectionParser(),
)

################################################################################

chunk_selection_template = MessageTemplate(
    template=[
        ("system", "{system_prompt}"),
        ("user", """
# Task
Your task is to analyse the providing context then decide which paragraph in the list may be useful for you to answer the given question. Select a most relevant paragraph from the given paragraph list.

# Output Format
Please output in following JSON format:
{{
    "thinking": <A string. Your thinking for this selection task.>,
    "paragraph_idx": <An integer. A paragraph index from 1 to {num_chunks}.>
}}

# Context
The context we already have:
{chosen_context}

# Paragraph List You Can Choose From
{chunk_list_str}

# Question
{content}

# Your output:
""".strip()),
    ],
    input_variables=["content", "chosen_context", "num_chunks", "chunk_list_str"],
    partial_variables={
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
)


class ChunkSelectionParser(BaseContentParser):
    def __init__(self) -> None:
        super().__init__()
        self._atom_info_candidates: List[AtomRetrievalInfo] = []

    def encode(
        self, content: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo], **kwargs,
    ) -> Tuple[str, dict]:
        context = atom_infos_to_context_string(chosen_atom_infos)

        chunk_list_str = ""
        for i, info in enumerate(atom_info_candidates):
            if info.source_chunk_title is not None:
                chunk_list_str += f"Paragraph {i + 1}: Title: {info.source_chunk_title}. Content: {info.source_chunk}\n"
            else:
                chunk_list_str += f"Paragraph {i + 1}: {info.source_chunk}\n"

        self._atom_info_candidates = atom_info_candidates

        return content, {
            "num_chunks": len(atom_info_candidates),
            "chosen_context": context,
            "chunk_list_str": chunk_list_str,
        }

    def decode(self, content: str, **kwargs) -> Tuple[bool, str, AtomRetrievalInfo]:
        try:
            output = parse_json(content)
            thinking: str = output["thinking"]
            paragraph_idx = output["paragraph_idx"]
            if paragraph_idx is not None and paragraph_idx > 0 and paragraph_idx <= len(self._atom_info_candidates):
                chosen_info = self._atom_info_candidates[paragraph_idx - 1]
                return True, thinking, chosen_info
            else:
                return False, thinking, None
        except Exception as e:
            print(f"[ChunkSelectionParser] content to decode: {content}")
            print(f"Exception: {e}")
            return False, "", None


chunk_selection_protocol = CommunicationProtocol(
    template=chunk_selection_template,
    parser=ChunkSelectionParser(),
)

################################################################################

class ContextQaParser(GenerationQaParser):
    def encode(self, content: str, chosen_atom_infos: List[AtomRetrievalInfo], **kwargs) -> Tuple[str, Dict]:
        _, supplementary =  super().encode(content, **kwargs)

        context_if_any = ""
        if len(chosen_atom_infos) > 0:
            context_if_any = atom_infos_to_context_string(chosen_atom_infos)
        supplementary["context_if_any"] = context_if_any

        return content, supplementary


final_qa_protocol = CommunicationProtocol(
    template=generation_qa_with_reference_template,
    parser=ContextQaParser(),
)
