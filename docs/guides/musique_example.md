# How to run MuSiQue example

In this document, we'll introduce how to run MuSiQue example step by step. To run experiments on open benchmarks like HotpotQA, 2WikiMultiHopQA, etc., steps are similar to those listed here. Please refer to this document and make some corresponding changes.

## Step 1. Testing suite preparation

For open benchmarks like MuSiQue, we have already prepared the preprocessing scripts, running the existing scripts is enough for the testing suite preparation. File *data_process/open_benchmarks/config/musique.yml* shows an example setting for MuSiQue dataset processing. In case you want to try other open benchmarks, the suggested split setting can be found in file *data_process/open_benchmarks/config/datasets.yaml*.

Assume that you are in the root directory of PIKE-RAG:

    ```sh
    # Install libraries required by the pre-processing
    pip install -r data_process/open_benchmarks/requirements.txt

    # Run script to download MuSiQue, sampling subset, transform format
    python data_process/main.py data_process/open_benchmarks/config/musique.yml
    ```

Once the script finishes, you can find the pre-processed data under *data/musique/*. Specifically, the sampled dataset is *data/musique/dev_500.jsonl*.

*Skip the part below and move to Step 3 if you only need to run MuSiQue.*

It should be noted that we only offers the preprocessing utils for a limited set of specific datasets (as listed in *data_process/open_benchmarks/dataset_utils/*). To preprocess other datasets, you can add the util functions referring to these utils.

Further more, for testing on specific domains, you need to preprocess the testing data in the following format to reuse the loading util functions we provided as in MuSiQue example, (or write a specific loading util function for your testing data and modify the setting when running qa script). The default testing suite file should be a `jsonlines` file, with each line presenting a dictionary of a question. The required field of Generation QA contains `"question" (type: str)` (and `"answer_labels" (type: List[str])` if automatically evaluation required). For Multiple-Choices QA, `"answer_mask_labels" (type: List[str])` is required, while `"answer_masks" (type: List[str])` is required for evaluation. Besides, you can maintain some other metadata in the field `"metadata" (type: dict)` as we did for the open benchmarks. To be specific, a general Generation QA would be in the format as below:

    ```py
    {
        "question": "Required, str. The question to be answered",
        "id": "Optional, str. The id of this question. For example, 'Q001' for easier reference.",
        "answer_labels": [
            "type: List[str]",
            "Required for automatic evaluation pipeline",
            "Length could be 1 or more than 1",
            "The answer labels used to calculate the metrics",
        ],
        "question_type": "Optional, str. The question type if useful. Could be set to 'undefined'",
        "metadata": {
            "meta_key": "Optional. Any other meta information",
        },
    }
    ```

## Step 2. Raw document preprocessing (Optional)

*Skip this Step and move to Step 3 if you only need to run MuSiQue.*

Sometimes the performance is highly related to how you pre-process the raw documents. In scenarios where raw documents are in the format of multi-modal like PDF documents and your application pursues extreme performance, we suggest you to leverage the Document Intelligence (DI) tools (e.g., [Azure AI Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)) to pre-process the raw documents.

Currently, we didn't provide scripts or components to integrate DI tools in PIKE-RAG. Please build up the DI-preprocessing pipeline according to your need.

## Step 3. Splitting original documents into chunks

To reproduce the experiments in the [technical report](https://arxiv.org/abs/2501.11551), there is no need to run the chunking script. Instead, we extract the context paragraphs of these open benchmarks and aggregate them together as the reference chunk pool. File *data_process/retrieval_contexts_as_chunks.py* shows an example to extract the context paragraphs.

Assume that you are in the root directory of PIKE-RAG:

    ```sh
    # Run script to extract the context paragraphs from the QA data.
    python data_process/retrieval_contexts_as_chunks.py
    ```

Once the script finished, you can find file *dev_500_retrieval_contexts_as_chunks.jsonl* under *data/musique/*.

*Skip the part below and move to Step 4 if you only need to run MuSiQue.*

File *examples/biology/configs/chunking.yml* is an example yaml config to leverage the context-aware document chunking to split markdown files into chunks. To run chunking task like this, you can modify the configuration according to your need and run the command:

    ```sh
    # Read in documents and split them into chunks.
    python examples/chunking.py PATH-TO-YAML-CONFIG
    ```

If you want to use lighter splitter without calling LLM models, we also offer a `RecursiveSentenceSplitter`. You can modify the `splitter` setting part in the yaml config to use it:

    ```yaml
    splitter:
        module_path: pikerag.document_transformers
        class_name: RecursiveSentenceSplitter
        args:
            ...  # Configure according to your need
    ```

We also supporting existing third party Splitters like `langchain.text_splitter.TextSplitter`. To use it, modify the yaml config file to:

    ```yaml
    splitter:
        module_path: langchain.text_splitter
        class_name: TextSplitter
        args:
            ...  # Configure according to your need
    ```

## Step 4. Atomic Question Tagging (Optional)

In current release version and in this MuSiQue example, we show a distillation method -- atomic question tagging. To tag atomic questions to MuSiQue sample set, assume that you are in the root directory of PIKE-RAG:

    ```sh
    python examples/tagging.py examples/musique/configs/tagging.yml
    ```

Once running finishes, you can find file *dev_500_retrieval_contexts_as_chunks_with_atom_questions.jsonl* under *data/musique/*.

## Step 5. Question Answering

For testing suite with `answer_labels`, evaluation can be done along with the question answering. To run retrieval based on the tagged atomic questions on MuSiQue, assume that you are in the root directory of PIKE-RAG:

    ```sh
    python examples/qa.py examples/musique/configs/atomic_decompose.yml
    ```

Once the running finishes, you can find the answer data in file `logs/musique/atomic_decompose/atomic_decompose.jsonl`, where each line corresponds to one QA `dict` data, with a new `answer (type: str)` field and an `answer_metadata (type: dict)` field.

If you want to test different algorithms, adjust the answer flow in Workflow and config it in *yaml file*. If the `answer_labels` are not ready in your testing suite, you can simply remove the `evaluator` part in the yaml config file to run question answering without evaluation.

## Step 6. Evaluation Only (Optional)

*Skip the part below if you only need to run MuSiQue.*

To run the evaluation workflow for answer logging `jsonlines` file following the format as PIKE-RAG generated, modify the *examples/evaluate.yml* file or create a new one referring to it, assume that you are in the root directory of PIKE-RAG:

    ```sh
    python examples/evaluate.py examples/evaluate.yml
    ```
