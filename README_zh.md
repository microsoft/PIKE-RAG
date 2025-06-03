<p align="center">
    <img src="./docs/source/images/logo/PIKE-RAG_horizontal_black-font.svg" alt="PIKE-RAG" style="width: 80%; max-width: 100%; height: auto;">
</p>

<p align="center">
    <a href="https://pike-rag.azurewebsites.net/">🌐在线演示</a>
    <a href="https://arxiv.org/abs/2501.11551">📊技术报告</a>
</p>

[![License](https://img.shields.io/github/license/microsoft/PIKE-RAG)](https://github.com/microsoft/PIKE-RAG/blob/main/LICENSE)
[![CodeQL](https://github.com/microsoft/PIKE-RAG/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/microsoft/PIKE-RAG/actions/workflows/github-code-scanning/codeql)
[![Release](https://img.shields.io/github/v/release/microsoft/PIKE-RAG)](https://github.com/microsoft/PIKE-RAG/releases)
[![ReleaseDate](https://img.shields.io/github/release-date-pre/microsoft/PIKE-RAG)](https://github.com/microsoft/PIKE-RAG/releases)
[![Commits](https://img.shields.io/github/commits-since/microsoft/PIKE-RAG/latest/main)](https://github.com/microsoft/PIKE-RAG/commits/main)
[![Pull Requests](https://img.shields.io/github/issues-pr/microsoft/PIKE-RAG)](https://github.com/microsoft/PIKE-RAG/pulls)
[![Issues](https://img.shields.io/github/issues/microsoft/PIKE-RAG)](https://github.com/microsoft/PIKE-RAG/issues)

# PIKE-RAG: 专业知识与推理增强生成系统

## 为什么选择 PIKE-RAG？

近年来，检索增强生成（RAG）系统在通过外部检索扩展大型语言模型（LLM）能力方面取得了显著进展。然而，这些系统在满足现实工业应用的复杂多样需求方面仍面临挑战。仅仅依赖直接检索不足以从专业语料库中提取深层领域特定知识并进行逻辑推理。为了解决这个问题，我们提出了PIKE-RAG（专业知识与推理增强生成）方法，该方法专注于提取、理解和应用领域特定知识，同时构建连贯的推理逻辑，逐步引导LLM得出准确的回答。

<p align="center">
    <img src="docs/source/images/readme/pipeline.png" alt="PIKE-RAG框架概览" style="width: 80%; max-width: 100%; height: auto;">
</p>

PIKE-RAG框架主要由几个基础模块组成，包括文档解析、知识提取、知识存储、知识检索、知识组织、以知识为中心的推理，以及任务分解与协调。通过调整主要模块内的子模块，可以实现专注于不同能力的RAG系统，以满足现实场景的多样化需求。

例如，在*患者历史医疗记录搜索*案例中，它专注于*事实信息检索能力*。主要挑战是：(1) 知识的理解和提取往往受到不当知识分割的阻碍，破坏了语义连贯性，导致检索过程复杂且低效；(2) 常用的基于嵌入的知识检索受到嵌入模型对齐专业术语和别名能力的限制，降低了系统准确性。通过PIKE-RAG，我们可以在知识提取过程中使用上下文感知分割技术、自动术语标签对齐技术和多粒度知识提取方法来提高知识提取和检索的准确性，从而增强事实信息检索能力，如下面的流水线所示。

<p align="center">
    <img src="docs/source/images/readme/L1_pipeline.png" alt="专注于事实信息检索的流水线" style="width: 80%; max-width: 100%; height: auto;">
</p>

对于像*为患者提供合理治疗方案和应对措施建议*这样的复杂任务，需要更高级的能力：需要强大的领域特定知识来准确理解任务，有时还需要合理地分解任务；还需要先进的数据检索、处理和组织技术来进行潜在趋势预测；多智能体规划也将有助于同时考虑创造性和可靠性。在这种情况下，可以初始化下面更丰富的流水线来实现这一目标。

<p align="center">
    <img src="docs/source/images/readme/L4_pipeline.png" alt="专注于基于事实的创新和生成的流水线" style="width: 80%; max-width: 100%; height: auto;">
</p>

在公共基准测试中，PIKE-RAG在HotpotQA、2WikiMultiHopQA和MuSiQue等多个多跳问答数据集上表现出色。与现有基准方法相比，PIKE-RAG在准确率和F1分数等指标上表现优异。在HotpotQA数据集上，PIKE-RAG达到了87.6%的准确率，在2WikiMultiHopQA上达到了82.0%，在更具挑战性的MuSiQue数据集上达到了59.6%。这些结果表明，PIKE-RAG在处理复杂推理任务方面具有显著优势，特别是在需要整合多源信息并进行多步推理的场景中。

PIKE-RAG已在工业制造、采矿和制药等领域进行了测试，并显著提高了问答准确性。未来，我们将继续探索其在更多领域的应用。此外，我们还将继续探索其他形式的知识和逻辑，以及它们对特定场景的最佳适应性。

## 更多详情

- 📊 [技术报告](https://arxiv.org/abs/2501.11551) 将阐述工业RAG问题分类，介绍PIKE-RAG的主要组件，并展示在公共基准测试中的一些实验结果。
- 🌐 [在线演示](https://pike-rag.azurewebsites.net/) 是我们针对L2 RAG任务的知识感知分解流水线的展示。

## 快速开始

1. 克隆此仓库并设置Python环境，参考[此文档](docs/guides/environment.md)；
2. 创建`.env`文件保存您的端点信息（以及其他需要的环境变量），参考[此文档](docs/guides/env_file.md)；
3. 修改*yaml配置*文件并尝试运行*examples/*目录下的脚本，参考[此文档](docs/guides/examples.md)；
4. 构建您自己的流水线和/或添加您自己的组件！

🚀 [此处](docs/guides/musique_example.md)提供了快速重现技术报告中MuSiQue实验的文档！

## 贡献

本项目欢迎贡献和建议。大多数贡献需要您同意贡献者许可协议（CLA），声明您有权并实际授予我们使用您贡献的权利。详情请访问 https://cla.opensource.microsoft.com。

当您提交拉取请求时，CLA机器人将自动确定您是否需要提供CLA并适当地装饰PR（例如，状态检查、评论）。只需按照机器人提供的说明操作即可。您只需要在使用我们CLA的所有仓库中执行一次此操作。

本项目采用了[Microsoft开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
更多信息请参阅[行为准则FAQ](https://opensource.microsoft.com/codeofconduct/faq/)或
联系[opencode@microsoft.com](mailto:opencode@microsoft.com)获取任何其他问题或评论。

## 商标

本项目可能包含项目、产品或服务的商标或徽标。Microsoft商标或徽标的授权使用须遵守并必须遵循[Microsoft商标和品牌指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。在本项目的修改版本中使用Microsoft商标或徽标不得引起混淆或暗示Microsoft赞助。任何第三方商标或徽标的使用都受这些第三方政策的约束。 