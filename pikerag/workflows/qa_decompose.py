# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple

from pikerag.knowledge_retrievers.chunk_atom_retriever import AtomRetrievalInfo, ChunkAtomRetriever
from pikerag.utils.config_loader import load_protocol
from pikerag.utils.logger import Logger
from pikerag.workflows.common import BaseQaData
from pikerag.workflows.qa import QaWorkflow


class QaDecompositionWorkflow(QaWorkflow):
    def __init__(self, yaml_config: Dict) -> None:
        super().__init__(yaml_config)

        workflow_configs: dict = self._yaml_config["workflow"].get("args", {})
        self._max_num_question: int = workflow_configs.get("max_num_question", 5)
        self._question_similarity_threshold: float = workflow_configs.get("question_similarity_threshold", 0.9)

    def _init_protocol(self) -> None:
        decompose_proposal_config = self._yaml_config["decompose_proposal_protocol"]
        self._decompose_proposal_protocol = load_protocol(
            module_path=decompose_proposal_config["module_path"],
            protocol_name=decompose_proposal_config["protocol_name"],
            partial_values=decompose_proposal_config.get("template_partial", {}),
        )

        retrieval_info_selection_config = self._yaml_config["selection_protocol"]
        self._retrieval_info_selection_protocol = load_protocol(
            module_path=retrieval_info_selection_config["module_path"],
            protocol_name=retrieval_info_selection_config["protocol_name"],
            partial_values=retrieval_info_selection_config.get("template_partial", {}),
        )

        backup_retrieval_info_selection_config = self._yaml_config.get("backup_selection_protocol", None)
        if backup_retrieval_info_selection_config is None or len(backup_retrieval_info_selection_config) == 0:
            self._backup_retrieval_info_selection_protocol = None
        else:
            self._backup_retrieval_info_selection_protocol = load_protocol(
                module_path=backup_retrieval_info_selection_config["module_path"],
                protocol_name=backup_retrieval_info_selection_config["protocol_name"],
                partial_values=backup_retrieval_info_selection_config.get("template_partial", {}),
            )

        original_question_answering_config = self._yaml_config["original_question_answering_protocol"]
        self._original_question_answering_protocol = load_protocol(
            module_path=original_question_answering_config["module_path"],
            protocol_name=original_question_answering_config["protocol_name"],
            partial_values=original_question_answering_config.get("template_partial", {}),
        )

    def _init_retriever(self) -> None:
        super()._init_retriever()
        assert isinstance(self._retriever, ChunkAtomRetriever)

        self._filter_logger = Logger("similarity_filter", dump_folder=self._yaml_config["log_dir"])

    def _propose_question_decomposition(
        self, question: str, chosen_atom_infos: List[AtomRetrievalInfo],
    ) -> Tuple[bool, str, List[str]]:
        """Let the LLM client propose a sub-questions list for better answering the original question based on the
        chosen information. It is the first step in atom information selection loop.
        """
        self._logger.debug("🤔 [问题分解调用] 准备调用LLM进行问题分解")
        self._logger.debug(f"🤔 [问题分解输入] 原始问题: {question}")
        self._logger.debug(f"🤔 [问题分解输入] 已有信息数量: {len(chosen_atom_infos)}")
        
        messages = self._decompose_proposal_protocol.process_input(
            content=question,
            chosen_atom_infos=chosen_atom_infos,
        )
        content = self._client.generate_content_with_messages(messages, **self.llm_config)
        decompose, thinking, question_list = self._decompose_proposal_protocol.parse_output(content)
        
        self._logger.debug(f"🤔 [问题分解输出] 分解决定: {decompose}")
        if decompose:
            self._logger.debug(f"🤔 [问题分解输出] 生成子问题数量: {len(question_list)}")
        
        return decompose, thinking, question_list

    def _filter_atom_infos(
        self, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo],
    ) -> List[AtomRetrievalInfo]:
        """Filter the atom information candidates based on the information we already chosen before. Currently we only
        filter out the atom information linked to same source chunk here.
        """
        if len(chosen_atom_infos) == 0:
            return atom_info_candidates

        # Filter out candidate if same chunk
        chosen_chunk_id = set([chosen_info.source_chunk_id for chosen_info in chosen_atom_infos])
        remaining_candidates = []
        for candidate in atom_info_candidates:
            if candidate.source_chunk_id in chosen_chunk_id:
                self._filter_logger.debug(
                    f"\n[filtered Atom] {candidate.atom}"
                    f"\n[Due to same chunk already exist]"
                )
            else:
                remaining_candidates.append(candidate)

        return remaining_candidates

    def _retrieve_atom_info_candidates(
        self, atom_queries: List[str], query: str, chosen_atom_infos: List[AtomRetrievalInfo], retrieve_id: str,
    ) -> List[AtomRetrievalInfo]:
        """Retrieve the atom information candidates from vector stores. It's designed to use the `atom_queries` to
        retrieve atom information while the `query` would be used as back-up retrieval methods. It is the second step
        in atom information selection loop.
        """
        assert isinstance(self._retriever, ChunkAtomRetriever)

        self._logger.debug(f"🔍 [信息检索] 使用查询列表检索，查询数量: {len(atom_queries)}")
        for i, query_text in enumerate(atom_queries, 1):
            self._logger.debug(f"🔍 [查询 {i}] {query_text}")

        # Retrieve atom info through atom storage by atom queries.
        atom_info_candidates = self._retriever.retrieve_atom_info_through_atom(
            queries=atom_queries,
            retrieve_id=retrieve_id,
        )
        self._logger.debug(f"🔍 [主要检索] 通过子问题检索到 {len(atom_info_candidates)} 个候选")
        
        atom_info_candidates = self._filter_atom_infos(atom_info_candidates, chosen_atom_infos)
        self._logger.debug(f"🔍 [过滤后] 剩余 {len(atom_info_candidates)} 个候选")

        # Backup retrieval 1: retrieve atom info through atom storage by original query.
        if len(atom_info_candidates) == 0:
            self._logger.debug("🔍 [备用检索1] 使用原始问题检索")
            atom_info_candidates = self._retriever.retrieve_atom_info_through_atom(
                queries=query,
                retrieve_id=retrieve_id,
            )
            atom_info_candidates = self._filter_atom_infos(atom_info_candidates, chosen_atom_infos)
            self._logger.debug(f"🔍 [备用检索1] 检索到 {len(atom_info_candidates)} 个候选")

        # Backup retrieval 2: retrieve atom info through chunk storage directly by original query.
        if len(atom_info_candidates) == 0:
            self._logger.debug("🔍 [备用检索2] 直接从chunk存储检索")
            atom_info_candidates = self._retriever.retrieve_atom_info_through_chunk(query, retrieve_id)
            atom_info_candidates = self._filter_atom_infos(atom_info_candidates, chosen_atom_infos)
            self._logger.debug(f"🔍 [备用检索2] 检索到 {len(atom_info_candidates)} 个候选")

        return atom_info_candidates

    def _select_atom_question(
        self, question: str, atom_info_candidates: List[AtomRetrievalInfo], chosen_atom_infos: List[AtomRetrievalInfo],
    ) -> Tuple[bool, str, AtomRetrievalInfo]:
        """Given the original question to be answered and the atom information we already have, let the LLM select the
        atom information that can best help the question answering.
        """
        assert len(atom_info_candidates) > 0, "Info candidate list is empty!"
        
        self._logger.debug(f"🎯 [信息选择] 从 {len(atom_info_candidates)} 个候选中选择")
        self._logger.debug(f"🎯 [信息选择] 已有信息数量: {len(chosen_atom_infos)}")
        
        messages = self._retrieval_info_selection_protocol.process_input(
            content=question,
            atom_info_candidates=atom_info_candidates,
            chosen_atom_infos=chosen_atom_infos,
        )
        content = self._client.generate_content_with_messages(messages, **self.llm_config)
        selected, thinking, chosen_atom = self._retrieval_info_selection_protocol.parse_output(content)

        self._logger.debug(f"🎯 [主要选择] 选择结果: {selected}")
        
        if not selected and self._backup_retrieval_info_selection_protocol is not None:
            self._logger.debug("🎯 [备用选择] 使用备用选择协议")
            messages = self._backup_retrieval_info_selection_protocol.process_input(
                content=question,
                atom_info_candidates=atom_info_candidates,
                chosen_atom_infos=chosen_atom_infos,
            )
            content = self._client.generate_content_with_messages(messages, **self.llm_config)
            selected, thinking2, chosen_atom = self._backup_retrieval_info_selection_protocol.parse_output(content)
            thinking += "\n" + thinking2
            self._logger.debug(f"🎯 [备用选择] 选择结果: {selected}")

        if selected:
            self._logger.debug(f"🎯 [选中信息] {chosen_atom.atom[:100]}...")

        return selected, thinking, chosen_atom

    def _answer_original_question(self, question: str, chosen_atom_infos: List[AtomRetrievalInfo]) -> Dict[str, str]:
        """Given the atom information we chosen, let the LLM answer the question.
        """
        messages = self._original_question_answering_protocol.process_input(
            content=question,
            chosen_atom_infos=chosen_atom_infos,
        )
        response = self._client.generate_content_with_messages(messages, **self.llm_config)
        output = self._original_question_answering_protocol.parse_output(response)
        if "response" not in output:
            output["response"] = response
        return output

    def answer(self, qa: BaseQaData, question_idx: int) -> Dict:
        """Decompose the question in the given qa step-by-step and give the answer to it in the end.

        Before give the answer to the question, there would be a decompose-retrieve-select loop to collect the useful
        atom information. In every loop, there are three steps:
        - Step 1: Proposal. Decompose the question given the atom information we already have. The output of step 1
            would be a list of sub-questions that may be useful to answer the final question;
        - Step 2: Retrieval. Retrieve the relevant atom information (including the atoms and source chunks) from vector
            stores with the sub-questions list we got in step 1 and the final question we are going to answer. The
            output of step 2 would be a list of atom information candidates.
        - Step 3: Selection. Select the most useful atom information from the given candidates, based on the final
            question to be answered and the atom information we already have. The output of step 3 would be the chosen
            atom information, if any.

        The final step out of the loop is to let the LLM answer the original question given all the chosen atom
        information.
        """
        # 开始问答流程日志
        self._logger.info("🚀" + "=" * 100)
        self._logger.info(f"🚀 [问答流程开始] 原始问题: {qa.question}")
        self._logger.info(f"🚀 [问答流程开始] 最大子问题数量: {self._max_num_question}")
        self._logger.info("🚀" + "=" * 100)
        
        decomposition_infos: dict = {}
        chosen_atom_infos: List[AtomRetrievalInfo] = []
        
        while len(chosen_atom_infos) < self._max_num_question:
            sub_question_id: str = f"Sub{len(chosen_atom_infos) + 1}"
            decomposition_infos[sub_question_id] = {}
            
            # 轮次开始日志
            self._logger.info("")
            self._logger.info("🔄" + "-" * 80)
            self._logger.info(f"🔄 [第 {len(chosen_atom_infos) + 1} 轮分解] 开始处理")
            self._logger.info(f"🔄 [当前状态] 已选择信息数量: {len(chosen_atom_infos)}")
            self._logger.info("🔄" + "-" * 80)

            # Step 1: Let LLM client provide a decomposition proposal with current context.
            self._logger.info("📝 [步骤1: 问题分解] 开始...")
            decompose, thinking, proposals = self._propose_question_decomposition(qa.question, chosen_atom_infos)
            
            # 分解结果日志
            self._logger.info(f"📝 [问题分解结果] 是否需要分解: {decompose}")
            if decompose:
                self._logger.info(f"📝 [问题分解推理] {thinking}")
                self._logger.info("📝 [提议的子问题列表]:")
                for i, proposal in enumerate(proposals, 1):
                    self._logger.info(f"   {i}. {proposal}")
            else:
                self._logger.info("📝 [问题分解] 无需进一步分解，退出循环")
            
            decomposition_infos[sub_question_id]["proposal"] = {
                "to_decompose": decompose,
                "thinking": thinking,
                "proposal_list": proposals,
            }
            if not decompose:
                break

            # Step 2: Retrieve relevant atom information to the sub-question proposals.
            self._logger.info("🔍 [步骤2: 信息检索] 开始...")
            self._logger.info(f"🔍 [检索查询] 使用 {len(proposals)} 个子问题进行检索")
            
            atom_info_candidates = self._retrieve_atom_info_candidates(
                atom_queries=proposals,
                query=qa.question,
                chosen_atom_infos=chosen_atom_infos,
                retrieve_id=sub_question_id,
            )
            
            # 检索结果日志
            self._logger.info(f"🔍 [检索结果] 找到 {len(atom_info_candidates)} 个候选信息")
            for i, info in enumerate(atom_info_candidates, 1):
                self._logger.info(f"   候选 {i}: {info.atom[:100]}...")
                self._logger.info(f"   来源: {info.source_chunk_title}")
            
            decomposition_infos[sub_question_id]["retrieval"] = [
                {
                    "relevant_proposal": info.atom_query,
                    "sub-question": info.atom,
                    "relevant_context_title": info.source_chunk_title,
                    "relevant_context": info.source_chunk,
                }
                for info in atom_info_candidates
            ]
            if len(atom_info_candidates) == 0:
                self._logger.info("🔍 [检索结果] 未找到相关信息，退出循环")
                break

            # Step 3: Let LLM client select following sub-question from the candidates with current context.
            self._logger.info("🎯 [步骤3: 信息选择] 开始...")
            selected, thinking, chosen_info = self._select_atom_question(
                qa.question,
                atom_info_candidates,
                chosen_atom_infos,
            )
            
            # 选择结果日志
            self._logger.info(f"🎯 [选择结果] 是否选中: {selected}")
            self._logger.info(f"🎯 [选择推理] {thinking}")
            
            decomposition_infos[sub_question_id]["selection"] = {
                "selected": selected,
                "thinking": thinking,
            }
            
            if selected:
                chosen_atom_infos.append(chosen_info)
                self._logger.info("✅ [信息已选中]")
                self._logger.info(f"   问题: {chosen_info.atom}")
                self._logger.info(f"   来源: {chosen_info.source_chunk_title}")
                self._logger.info(f"   内容摘要: {chosen_info.source_chunk[:200]}...")
                
                decomposition_infos[sub_question_id]["selection"]["chosen_info"] = {
                    "question": chosen_info.atom,
                    "source_chunk_title": chosen_info.source_chunk_title,
                    "source_chunk": chosen_info.source_chunk,
                }
            else:
                self._logger.info("❌ [未选中信息] 退出循环")
                break
            
            self._logger.info(f"🔄 [第 {len(chosen_atom_infos)} 轮完成] 当前已选择 {len(chosen_atom_infos)} 个信息")

        # 多跳推理总结
        self._logger.info("")
        self._logger.info("🧠" + "=" * 100)
        self._logger.info(f"🧠 [多跳推理完成] 总共进行了 {len(chosen_atom_infos)} 轮信息收集")
        self._logger.info("🧠 [最终选择的信息摘要]:")
        for i, info in enumerate(chosen_atom_infos, 1):
            self._logger.info(f"   {i}. 问题: {info.atom}")
            self._logger.info(f"      来源: {info.source_chunk_title}")
        self._logger.info("🧠" + "=" * 100)

        # Last Step: Let LLM client answer the original question with all chosen atom information during the loop above.
        self._logger.info("📋 [最终答案生成] 开始...")
        self._logger.info(f"📋 [最终答案生成] 基于 {len(chosen_atom_infos)} 个选中的信息")
        
        output = self._answer_original_question(qa.question, chosen_atom_infos)
        
        # 最终答案日志
        if "answer" in output:
            self._logger.info("🎉 [最终答案生成完成]")
            self._logger.info("🎉" + "=" * 100)
            self._logger.info("🎉 [最终答案]:")
            self._logger.info(output["answer"])
            self._logger.info("🎉" + "=" * 100)
        
        output["decomposition_infos"] = decomposition_infos
        return output
