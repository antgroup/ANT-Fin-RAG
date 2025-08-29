# Mitigating Hallucination in Financial Retrieval-Augmented Generation via Fine-Grained Knowledge Verification

This repository primarily consists of two parts: our RLFKV code implementation and the Financial Data Fidelity Evaluation Dataset FDD_Hard, which includes various types of data such as stocks, macroeconomics, and funds.


## RLFKV Implementation Details
Our RLFKV addresses hallucination mitigation in financial text generation through fine-grained knowledge verification. The implementation is built on the [ms-swift](https://github.com/modelscope/ms-swift) framework, with the core contribution being the [RLFKV](./code/ms-swift-rlfkv/examples/train/grpo/plugin/plugin.py) function.

This method decomposes long-form responses into fine-grained atomic knowledge units and verifies the factual accuracy of each unit, providing granular signals to guide the model during reinforcement fine-tuning.

 **Once our paper is accepted, we will upload the code.**

### Installation
```
cd ms-swift-rlfkv
pip install -e .
```

### training
    bash scripts/train.sh

## evaluating 
    bash scripts/evaluate.sh

## FDD_Hard Details



The comparative analysis with BizFinBench's [FDD](https://github.com/HiThink-Research/BizFinBench/blob/main/datasets/Financial_Data_Description.jsonl) reveals the following advantages of FDD_Hard: Superior coverage (3 financial data types versus FDD's single type) and Enhanced complexity (13× longer average context length).

| Dimension          | FDD              | FDD_Hard         |
|--------------------|------------------|------------------|
| **Data Types**     | Stocks only      | Stocks, Macro, Funds |
| **Data Volume**    | 1,461 samples    | 2,000 samples    |
| **Avg. Length**    | 506 tokens       | 6,763 tokens     |




 


## Contact
For questions or feedback, please reach out to our team:
+ Taoye Yin (yintaoye.yty@antgroup.com)
+ Yaxin Fan (fanyaxin.fyx@antgroup.com)