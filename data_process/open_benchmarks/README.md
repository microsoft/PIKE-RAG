# Dataset Protocol

## Pipeline and Script Description

Assume that you are in the directory `data_process/`:

```sh
python main.py open_benchmarks/config/datasets.yaml
```

## QA Protocol Overview

We defined a general QA data protocol as below:

```py
qa: Dict = {
    # str: The unique qa id generated with uuid.uuid4().hex.
    "id": "a unique qa id",

    # str: The question to be answered.
    "question": "the question to be answered",

    # List[str]: A list of correct answers, could be a list of single value for some datasets.
    "answer_labels": ["a list of correct answers", "there could be only one answer for some datasets"],

    # Literal["yes_no", "undefined"]: may be extended in the future.
    #   "yes_no" indicates the answer should be in ["yes", "no"].
    "question_type": "undefined",

    # Dict[str, Union[str, List]]: The set of metadata information, varied among different datasets.
    "metadata": {
        # Union[str, int] if exists: The qa id defined in the original dataset.
        "original_id": "the id of type str or int if exists",

        # str if exists: The question type defined in the original dataset. Values varied among different datasets.
        "original_type": "bridge",

        # Literal["easy", "median", "hard"] if exists: The difficulty level defined in the original dataset.
        #   Currently, it only exists for HotpotQA.
        "original_level": "hard",

        # List[Dict]: The supporting facts that are useful to answer the question, if exists.
        #   It corresponds to the "supporting_facts"/"evidence_span"/"long_answers"/... in the original datasets.
        #   Refers to the introduction of each dataset for more details.
        "supporting_facts": [
            {
                # Literal["wikipedia", "wikidata", "BingSearch"]: The type of the supporting fact,
                #   may be extended in the future.
                #   Each type of supporting fact corresponds to a specific set of valid keys in the dict.
                "type": "wikipedia",

                # str: The supporting wikipedia title.
                "title": "wikipedia example 1",

                # str: The supporting wikipedia contents. May be a long string with multiple sentences.
                "contents": "wikipedia contents example 1",
            },
            {
                "type": "wikidata",

                # str: The supporting wikidata title.
                "title": "wikidata title example 1",

                # str: The supporting wikidata section.
                "section": "wikidata section example 1",

                # str: The supporting wikidata contents in the specified section.
                "contents": "wikidata contents example 1",
            },
        ],

        # List[Dict]: The retrieval contexts provided by the original dataset, if exists.
        #   It corresponds to the "contexts"/... in the original datasets.
        #   Refer to the introduction of each dataset for more details.
        "retrieval_contexts": [
            {
                # Literal["wikipedia", "wikidata", "BingSearch"]: The type of the retrieval contexts,
                #   may be extended in the future.
                #   Each type of retrieval contexts correspond to a specific set of valid keys in the dict.
                "type": "wikipedia",

                # str: The wikipedia title of the contexts.
                "title": "wikipedia example 2",

                # str: The wikipedia contexts. May be a long string with multiple sentences.
                "contents": "wikipedia contents example 2",
            },
            {
                "type": "wikidata",

                # str: The wikidata title of the contexts.
                "title": "wikidata title example 2",

                # str: the wikidata section of the contexts.
                "section": "wikidata section example 2",

                # str: the wikidata contents in the specified section.
                "contents": "wikidata contents example 2",
            },
            {
                "type": "BingSearch",

                # str: The title of the returned item.
                "title": "Bing search title example 1",

                # str: The url of the returned item.
                "url": "http://example.url.com",

                # str: The simple description showed in the search page.
                "description": "Bing search description example 1",

                # str: The contents of the returned item.
                "contents": "Bing search contents example 1",

                # int: The search rank of the item in the search results.
                "rank": 1,
            },
        ],
        "reasoning_logics": [
            {
                "type": "wikidata",

                # str: The supporting wikidata title.
                "title": "wikidata title example 3",

                # str: The supporting wikidata section.
                "section": "wikidata section example 3",

                # str: The supporting wikidata contents in the specified section.
                "contents": "wikidata contents example 3",
            },
        ]
    },
}
```

## Dataset Details

### Natural Questions (nq)

The original dataset was downloaded from [HuggingFace](https://huggingface.co/datasets/google-research-datasets/natural_questions
).

#### Natural Questions Metadata

```py
"metadata": {
    # str: The `id` defined in the original dataset.
    "original_id": "the id of type str",

    # Literal["yes_no, undefined"]: corresponds to the `yes_no_answer` in the original dataset,
    #   if not "yes_no_answer", the value will be "undefined" here.
    "original_type": "yes_no",

    # List[Dict]: The `annotations[long_answers]` defined in the original dataset.
    "supporting_facts": [
        {
            # Literal["wikipedia"]: The type of the supporting fact can only be wikipedia in Natural Questions.
            "type": "wikipedia",

            # str: The supporting wikipedia title,
            #   corresponds to the `document["title"]` in the original dataset.
            "title": "The Walking Dead (season 8)",

            # str: The supporting wikipedia contents,
            #   corresponds to the bytes in the indicated wikipage from `start token` to `end toekn` inside the `long answers` dict.
            "contents": "The eighth season of The Walking Dead, an American post-apocalyptic horror television series on AMC, premiered on October 22, 2017 ...",
        },
    ],
},
```

#### Natural Questions Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** |
|:--------------:|--------:|----------------------:|:------------------------:|
|    **train**   | 106,926 |           48,525      |             1            |
| **validation** |   3,156 |            2,883      |             1            |
|    **Total**   | 110,082 |           49,440      |             1            |

### Trivia QA (triviaqa)

The original dataset was downloaded from [HuggingFace](https://huggingface.co/datasets/mandarjoshi/trivia_qa).

#### Trivia QA metadata

```py
"metadata": {
    # str: The `question_id` defined in the original dataset.
    "original_id": "the id of type str",

    # List[Dict]: The `entity_pages` and `search_results` defined in the original dataset.
    "supporting_facts": [
        {
            # Literal["wikipedia", "BingSearch"]:
            #   "wikipedia" indicates that it is a kind of `entity_pages`, with valid keys: ["title"].
            "type": "wikipedia",

            # str: The supporting wikipedia title,
            #   corresponds to the `entity_pages["title"]` if list is not empty.
            "title": "England"
        },

        {
            # Literal["wikipedia", "BingSearch"]:
            #   "BingSearch" indicates that it is a kind of `search_results`, with valid keys: ["title", "url", "description", "contents"].
            "type": "BingSearch",

            # str: The search results listed in the browser by Bing.
            "title": "Our History | Fiji Airways",

            # str: Corresponds to the url to the webpage indicated by the title.
            "url": "http://www.fijiairways.com/about-fiji-airways/our-history/",

            # str: Corresponds to the summary for the webpage indicated by the title.
            "description": "Our History - Six Decades of ... Government to operate the country’s domestic airline and registered ... Air Pacific a world-class international airline. “Air ...",

            # str: Corresponds to the text in the webpage indicated by the title.
            "contents": "An example web content"
        }
    ],
},
```

#### Trivia QA Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** |
|:--------------:|--------:|----------------------:|:------------------------:|
|    **train**   | 138,384 |           44,127      |           4.62           |
| **validation** |  17,944 |           10,467      |           4.61           |
|    **Total**   | 156,328 |           47,470      |           4.61           |

### HotpotQA (hotpotqa)

It is a question answering dataset proposed by [this work](https://arxiv.org/abs/1809.09600) in 2018.
The original dataset was downloaded from [GitHub](https://hotpotqa.github.io/).

#### HotpotQA Metadata

```py
"metadata": {
    # str: The `_id` defined in the original dataset.
    "original_id": "the id of type str",

    # Literal["bridge", "comparison"]: The `type` defined in the original dataset.
    #   "bridge" indicates questions that require reasoning over multiple documents by identifying a bridge entity
    #     that connects the information needed to answer the question.
    #   "comparison" indicates questions that require comparing attributes or properties of two or more entities.
    "original_type": "bridge",

    # Literal["easy", "median", "hard"]: The `level` defined in the original dataset.
    #   "easy" indicates single-hop questions that can be answered by reasoning within a single paragraph.
    #   "medium" indicates multi-hop questions that require reasoning over multiple paragraphs or documents.
    #   "hard" indicates challenging multi-hop questions that require advanced reasoning skills,
    #     where even state-of-the-art models struggle to find the correct answers.
    "original_level": "hard",

    # List[Dict]: The `supporting_facts` defined in the original dataset.
    "supporting_facts": [
        {
            # Literal["wikipedia"]: The type of the supporting fact can only be wikipedia in HotpotQA.
            "type": "wikipedia",

            # str: The supporting wikipedia title,
            #   corresponds to the first item in the tuple in original dataset.
            "title": "Ed Wood (film)",

            # str: The supporting wikipedia contents,
            #   corresponds to the sentences indicated by the second item in the tuple in original dataset.
            "contents": "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.",
        },
    ],

    # List[Dict]: The `contexts` defined in the original dataset.
    "retrieval_contexts": [
        {
            # Literal["wikipedia"]: The type the retrieval context can only be wikipedia in HotpotQA.
            "type": "wikipedia",

            # str: The wikipedia title of the contexts,
            #   corresponds to the first item in the tuple in original dataset.
            "title": "Scott Derrickson",

            # str: The wikipedia contexts,
            #   corresponds to the second item (List[str]) in the tuple in original dataset,
            #     the sentences (List[str]) are joined here to construct a long string.
            "contents": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California.",
        },
    ],
},
```

#### Hotpot QA Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** | **Mean #RetrievalContexts** |
|:--------------:|--------:|----------------------:|:------------------------:|:---------------------------:|
|    **train**   |  94,007 |          482,021      |           2.38           |             9.95            |
|     **dev**    |   7,405 |           66,581      |           2.43           |             9.95            |
|    **Total**   | 101,405 |          507,494      |           2.40           |             9.95            |

### 2WikiMultihopQA (2wiki)

It is a multi-hop question answering dataset proposed in [this work](https://arxiv.org/abs/2011.01060) in 2020.
The original dataset was downloaded from [Dropbox](https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1).

#### 2WikiMultihopQA metadata

```py
"metadata": {
    # str: The `id` defined in the original dataset.
    "original_id": "the id of type str",

    # Literal["comparison", "inference", "compositional", "bridge_comparison"]: The `type` defined in the original dataset.
    #   "comparison" indicates a type of question that compares two or more entities from the same group in some aspects of the entity.
    #   "inference" indicates a type of question created from two triples (e, r1, e1) and (e1, r2, e2) in the knowledge base (KB).
    #     A logical rule is used to derive a new triple (e, r, e2), where r is the inference relation obtained from r1 and r2.
    #   "compositional" indicates a type of question created from two triples (e, r1, e1) and (e1, r2, e2) in the KB,
    #     but without an inference relation existing between r1 and r2.
    #   "bridge_comparison" indicates a type of question that combines bridge and comparison questions. It requires both
    #     finding the bridge entities that connect paragraphs and performing comparisons to derive the answer.
    "original_type": "bridge_comparison",

    # List[Dict]: The `supporting_facts` and `evidences` defined in the original dataset.
    "supporting_facts": [
        {
            # Literal["wikipedia"],
            #   wikipedia: corresponds to the `supporting_facts` in original dataset.
            "type": "wikipedia",

            # str: The supporting wikipedia title,
            #   corresponds to the first item in `supporting_facts` list.
            "title": "Polish-Russian War (film)",

            # str: The supporting wikipedia contents,
            #   corresponds to the sentences index indicated by the second item (int) in `supporting_facts` list.
            "contents": "Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.",
        },
    ],

    # List[Dict]: The `contexts` defined in the original dataset.
    "retrieval_contexts": [
        {
            # Literal["wikipedia"]: The type the retrieval context can only be wikipedia in 2WikiMultihopQA.
            "type": "wikipedia",

            # str: The wikipedia title of the contexts,
            #   corresponds to the first item inside every list under `context`.
            "title": "Alice Washburn",

            # str: The wikipedia contexts,
            #   corresponds to the second item (List[str]) inside every list under `context`, the sentences are joined
            #     here to construct a long string.
            "contents": "Alice Washburn( 1860- 1929) was an American stage and film actress. She worked at the Edison, Vitagraph and Kalem studios. Her final film Snow White was her only known feature film.",
        },
    ],

    # List[Dict]: The `evidences` defined in the original dataset.
    "reasoning_logics": [
        {
            "type": "wikidata",

            # str: The supporting wikidata title,
            #   corresponds to the first item in every `evidences` list.
            "title": "Wedding Night in Paradise",

            # str: the supporting wikidata section,
            #   corresponds to the second item in every `evidences` list.
            "section": "director",

            # str: The supporting wikidata contents,
            #   corresponds to the third item in every `evidences` list.
            "contents": "Géza von Bolváry",
        },
    ],
},
```

#### 2WikiMultihopQA Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** | **Mean #RetrievalContexts** |
|:--------------:|--------:|----------------------:|:------------------------:|:---------------------------:|
|    **train**   | 167,454 |          369,378      |           4.87           |              10             |
|     **dev**    |  12,576 |           54,957      |           4.91           |              10             |
|    **Total**   | 180,030 |          384,857      |           4.89           |              10             |

### Pop QA (popqa)

The original dataset was downloaded from [HuggingFace](https://huggingface.co/datasets/akariasai/PopQA).

#### Pop QA Metadata

```py
"metadata": {
    # str: The `id` defined in the original dataset.
    "original_id": "the id of type str",

    # List[Dict]: The `subj, prop, obj` defined in the original dataset.
    "supporting_facts": [
        {
            # Literal["wikidata"]: The type of the supporting fact can only be wikidata in Pop QA.
            "type": "wikidata",

            # str: The supporting wikidata title,
            #   corresponds to `subj` in the original dataset.
            "title": "George Rankin",

            # str: The supporting wikidata section,
            #   corresponds to `prop` in the original dataset.
            "section": "occupation",

            # str: The supporting wikidata contents,
            #   corresponds to the `obj` in the original dataset.
            "contents": "politician",
        },
    ],
},
```

#### Pop QA Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** |
|:--------------:|--------:|----------------------:|:------------------------:|
|     **test**   |  14,267 |           12,244      |             1            |

### Web Questions (webqa)

The original dataset was downloaded from [HuggingFace](https://huggingface.co/datasets/Stanford/web_questions)

#### Web QA metadata

```py
"metadata": {
    "supporting_facts": [
        {
            # Literal["wikipedia"]: The type of the supporting fact can only be wikipedia in webqa.
            #   Parsed from the `url` in the original dataset if corresponding wiki-page exists.
            "type": "wikipedia",

            # str: The supporting wikidata title, parsed from the `url` in the original dataset.
            "title": "justin bieber"
        },
    ],
},
```

#### Web QA Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** |
|:--------------:|--------:|----------------------:|:------------------------:|
|    **train**   |   3,778 |          1,846        |            1             |
|     **test**   |   2,032 |          1,200        |            1             |
|    **Total**   |   5,810 |          2,420        |            1             |

### MuSiQue: Multi-Hop Questions via Single-hop Question Composition (musique)

It is a question answering dataset with 2-4 hops proposed in [this work](https://arxiv.org/abs/2108.00573) in 2022.
The original dataset was downloaded from [Google Driver](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view).

#### MuSiQue Metadata

```py
"metadata": {
    # str: The `id` defined in the original dataset.
    "original_id": "the id of type str",

    # List[Dict]: The `question_decomposition` defined in the original dataset.
    "supporting_facts": [
        {
            # Literal["wikipedia"]: The type of the supporting fact can only be wikipedia in MusiQue.
            "type": "wikipedia",

            # str: The supporting wikipedia title,
            #   corresponds to the `title` inside the indicated dict under `paragraphs`, the dict index is given by
            #     `paragraph_support_idx` inside every dict under `question_decomposition`.
            "title": "Mike Medavoy",

            # str: The supporting wikipedia paragraphs,
            #   corresponds to the `paragraph_text` inside indicated dict with the same `paragraph_support_idx` as "title".
            "contents": "Morris Mike Medavoy (born January 21, 1941) is an American film producer and executive, co-founder of Orion Pictures (1978), former chairman of TriStar Pictures, former head of production for United Artists and current chairman and CEO of Phoenix Pictures.",
        },
    ],

    # List[Dict]: The `paragraphs` defined in the original dataset.
    "retrieval_contexts": [
        {
            # Literal["wikipedia"]: The type the retrieval context can only be wikipedia in Musique.
            "type": "wikipedia",

            # str: The wikipedia title of the contexts,
            #   corresponds to the `title` inside every dict under `paragraphs`.
            "title": "Green Lake (Chisago City, Minnesota)",

            # str: The wikipedia contexts,
            #   corresponds to the `paragraph_text` inside every dict under `paragraphs`, the sentences (List[str]) are
            #     joined here to construct a long string.
            "contents": "Green Lake is a lake in Chisago City, Minnesota, United States. This lake is sometimes also referred to as \"Big Green Lake\" because it is connected to Little Green Lake by a channel. Green Lake was named from the fact its waters are green from the high algae content.",
        },
    ],
},
```

#### Musique Statistics

|    **Split**   | **#QA** | **Unique #Documents** | **Mean #SupportingFact** | **Mean #RetrievalContexts** |
|:--------------:|--------:|----------------------:|:------------------------:|:---------------------------:|
|    **train**   |  19,938 |           71,090      |           2.33           |              20             |
|     **dev**    |   2,147 |           17,629      |           2.64           |             19.99           |
|    **Total**   |  22,085 |           84,459      |           2.49           |              20             |
