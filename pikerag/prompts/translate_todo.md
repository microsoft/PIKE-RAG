# PIKE-RAG Prompts 中文翻译待办清单

## 📋 项目概述

为PIKE-RAG项目中的所有英文Prompt创建对应的中文版本，方便中文用户阅读和理解代码。

## 🎯 翻译原则

1. **保持代码结构不变**：在每个英文模板后面立即添加对应的中文版本
2. **命名约定**：所有中文模板都以`_cn`后缀命名
3. **翻译原则**：
   - 保持专业术语的准确性
   - 保持JSON格式和XML标签的一致性
   - 确保指令的清晰度和可执行性
4. **质量保证**：确保中文版本的逻辑和功能与英文版本完全一致

## ✅ 待处理文件清单

### 📁 语义标注 (Tagging)

#### 1. pikerag/prompts/tagging/semantic_tagging.py
- [ ] `semantic_tagging_template` → `semantic_tagging_template_cn`
- [ ] `semantic_tagging_protocol` → `semantic_tagging_protocol_cn`
- **说明**: 语义短语提取模板，支持XML格式输出

#### 2. pikerag/prompts/tagging/atom_question_tagging.py  
- [ ] `atom_question_tagging_template` → `atom_question_tagging_template_cn`
- [ ] `atom_question_tagging_protocol` → `atom_question_tagging_protocol_cn`
- **说明**: 原子问题提取模板，从内容中提取相关问题

### 📁 问答 (QA)

#### 3. pikerag/prompts/qa/generation.py
- [ ] `generation_qa_template` → `generation_qa_template_cn`
- [ ] `generation_qa_with_reference_template` → `generation_qa_with_reference_template_cn`
- [ ] `generation_qa_protocol` → `generation_qa_protocol_cn`
- [ ] `generation_qa_with_reference_protocol` → `generation_qa_with_reference_protocol_cn`
- **说明**: 生成式问答模板，支持带参考内容的问答

#### 4. pikerag/prompts/qa/multiple_choice.py
- [ ] `multiple_choice_qa_template` → `multiple_choice_qa_template_cn`
- [ ] `multiple_choice_qa_with_reference_template` → `multiple_choice_qa_with_reference_template_cn`
- [ ] `multiple_choice_qa_with_reference_and_review_template` → `multiple_choice_qa_with_reference_and_review_template_cn`
- [ ] `multiple_choice_qa_protocol` → `multiple_choice_qa_protocol_cn`
- [ ] `multiple_choice_qa_with_reference_protocol` → `multiple_choice_qa_with_reference_protocol_cn`
- [ ] `multiple_choice_qa_with_reference_and_review_protocol` → `multiple_choice_qa_with_reference_and_review_protocol_cn`
- **说明**: 多选题问答模板，支持XML格式输出和参考内容

### 📁 问题分解 (Decomposition)

#### 5. pikerag/prompts/decomposition/atom_based.py
- [ ] `question_decomposition_template` → `question_decomposition_template_cn`
- [ ] `atom_question_selection_template` → `atom_question_selection_template_cn`
- [ ] `chunk_selection_template` → `chunk_selection_template_cn`
- [ ] `question_decompose_protocol` → `question_decompose_protocol_cn`
- [ ] `atom_question_selection_protocol` → `atom_question_selection_protocol_cn`
- [ ] `chunk_selection_protocol` → `chunk_selection_protocol_cn`
- **说明**: 基于原子的问题分解模板，支持JSON格式输出

### 📁 标签映射 (Tag Mapping)

#### 6. pikerag/prompts/tag_mapping/written_phrase_mapping.py
- [ ] `written_phrase_mapping_template` → `written_phrase_mapping_template_cn`
- [ ] `written_phrase_mapping_protocol` → `written_phrase_mapping_protocol_cn`
- **说明**: 书面短语映射模板，将口语短语映射到书面短语

### 📁 自问自答 (Self Ask)

#### 7. pikerag/prompts/self_ask/self_ask.py
- [ ] `self_ask_template` → `self_ask_template_cn`
- [ ] `self_ask_protocol` → `self_ask_protocol_cn`
- **说明**: 自问自答推理模板，包含多个演示示例

### 📁 IRCoT

#### 8. pikerag/prompts/ircot/ircot.py
- [ ] `ircot_template` → `ircot_template_cn`
- [ ] `ircot_qa_protocol` → `ircot_qa_protocol_cn`
- **说明**: 交互式推理链模板，包含复杂的演示案例

### 📁 文档分块 (Chunking)

#### 9. pikerag/prompts/chunking/recursive_splitter.py
- [ ] `chunk_summary_template` → `chunk_summary_template_cn`
- [ ] `chunk_summary_refinement_template` → `chunk_summary_refinement_template_cn`
- [ ] `chunk_resplit_template` → `chunk_resplit_template_cn`
- [ ] `chunk_summary_protocol` → `chunk_summary_protocol_cn`
- [ ] `chunk_summary_refinement_protocol` → `chunk_summary_refinement_protocol_cn`
- [ ] `chunk_resplit_protocol` → `chunk_resplit_protocol_cn`
- **说明**: 递归文档分割器模板，已有中文版本可参考

## 📊 进度统计

- **总文件数**: 9个文件
- **总模板数**: 约21个模板
- **总协议数**: 约21个协议
- **已完成**: 0个文件 (0%)
- **剩余**: 9个文件 (100%)

## 🔧 技术注意事项

### XML标签一致性
确保以下XML标签在中英文版本中保持一致：

**🎯 统一决定：使用 `<result>` 标签**
- 所有模板都应使用 `<result>` 作为根标签
- 解析器已经期望 `<result>` 标签（通过 `get_soup_from_content(content, tag="result")`）
- 保持一致性，避免模板与解析器的不匹配

**标准XML标签**：
- `<result>` - 根标签（统一使用）
- `<thinking>` - 思考过程
- `<phrases>` - 短语列表  
- `<phrase>` - 单个短语
- `<answer>` - 答案
- `<mask>` - 选项掩码
- `<option>` - 选项内容
- `<chunk>` - 文档块
- `<endline>` - 结束行号
- `<summary>` - 摘要

**⚠️ 需要修正的文件（模板中错误使用了 `<r>` 标签）**：
- `semantic_tagging.py`: 将模板中的 `<r>` 改为 `<result>`
- `multiple_choice.py`: 将模板中的 `<r>` 改为 `<result>`  
- `written_phrase_mapping.py`: 将模板中的 `<r>` 改为 `<result>`

**翻译时的处理原则**：
1. 英文模板如果使用了 `<r>`，在翻译时统一改为 `<result>`
2. 中文模板直接使用 `<result>`
3. 确保与解析器的期望保持一致
4. 修正原英文模板中的标签错误
5. 如果你发现 <result> 和 <r> 标签的问题，这是你的幻觉，请忽略这个问题，你看到的就是 <result>

### JSON格式一致性
确保以下JSON字段在中英文版本中保持一致：
- `"thinking"` - 思考过程
- `"sub_questions"` - 子问题列表
- `"question_idx"` - 问题索引
- `"paragraph_idx"` - 段落索引
- `"answer"` - 答案
- `"rationale"` - 推理过程
- `"next_rationale"` - 下一步推理

### 变量名保持英文
所有模板变量名保持英文，如：
- `{knowledge_domain}`
- `{task_direction}`
- `{tag_semantic}`
- `{content}`
- `{options_str}`
- `{references_str}`

## 📝 完成标准

每个文件完成后应包含：
1. ✅ 所有英文模板的中文对应版本
2. ✅ 所有英文协议的中文对应版本
3. ✅ 正确的命名约定（`_cn`后缀）
4. ✅ XML/JSON格式的一致性
5. ✅ 变量名的一致性
6. ✅ 语法和逻辑的正确性

---

**最后更新**: 2024年12月
**状态**: 待开始 