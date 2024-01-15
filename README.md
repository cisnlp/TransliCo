# TransliCo

This is the repository for ***TransliCo*** framework, which aims to fine-tune an mPLM by contrasting sentences in its training data and their transliterations in a unified script (**Latn**, in our case). The framework therefore aligns sentences in their original scripts with their transliterations, which ensures uniformity in the representation space for different scripts. We use [Glot500](https://github.com/cisnlp/Glot500), a PLM pretrained on over 500 languages, as our source model, and find-tune it on a small portion (5%) of its training data: Glot500-c. The resulting model is referred to as **Furina**. This repo is based on [Glot500](https://github.com/cisnlp/Glot500) and [OFA](https://github.com/cisnlp/ofa).

Paper on arXiv: https://arxiv.org/abs/2401.06620

```
.
├── ContinuedPretrainerSingle.py
├── README.md
├── evaluation
│   ├── retrieval
│   │   ├── bible_lang_list.txt
│   │   ├── evaluate_retrieval_all_tatoeba.py
│   │   ├── evaluate_retrieval_all_tatoeba.sh
│   │   ├── evaluate_retrieval_bible.py
│   │   ├── evaluate_retrieval_bible_xlm.sh
│   │   └── tatoeba_lang_list.txt
│   ├── tagging
│   │   ├── evaluate_all_ner.py
│   │   ├── evaluate_all_ner.sh
│   │   ├── evaluate_all_pos.py
│   │   ├── evaluate_all_pos.sh
│   │   ├── ner_lang_list.txt
│   │   ├── pos_lang_list.txt
│   │   ├── run_tag.py
│   │   └── utils_tag.py
│   └── taxi1500
│       ├── evaluate_all.py
│       ├── evaluate_all.sh
│       └── texi1500_lang_list.txt
├── model_architecture.py
├── preprocess_dataset.py
├── requirements.txt
├── run_finetune.py
├── run_finetune.sh
├── uroman.py
└── utils.py
```

## Transliteration Data Generation

First concatenate the sentences from all language-scripts into a single file. Then use the following command to transiliterate it and create a csv file where each row is a pair of sentences (in its original script and in the Latn script). 

```
python preprocess_dataset.py
```


## Fine-tuning on the Paired Data

To fine-tune the model on the paired data generated above, run the following command:  

```
bash run_finetune.sh
```


## Model Loading

We release **Furina** and **Furina<sub>Indic</sub>** on Huggingface, you can download [Furina](https://huggingface.co/yihongLiu/furina) and [Furina<sub>indic</sub>](https://huggingface.co/yihongLiu/furina-indic).


To use **Furina** and **Furina<sub>indic</sub>**, you could simply load it through pipeline:

```python
>>> from transformers import pipeline
>>> MODEL_PATH = 'your_saved_model_path'
>>> mask_filler = pipeline('fill-mask', model=MODEL_PATH)
>>> mask_filler("Hello I'm a <mask> model.", tok_k=3)
``` 

or

```python
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer

MODEL_PATH = 'your_saved_model_path'

model = XLMRobertaForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)


text = "Hello I'm a <mask> model."
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()


for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

``` 

## Evaluation

### Dataset Preparation

Please refer to [Glot500](https://github.com/cisnlp/Glot500) for downloading the datasets used for evaluation.

### Sentence Retrieval - Bible

For SR-B, first go to ``evaluation/retrieval`` and run:

```
bash evaluate_retrieval_bible_xlm.sh
```


### Sentence Retrieval - Tatoeba

For SR-T, first go to ``evaluation/retrieval`` and run:

```
bash evaluate_retrieval_all_tatoeba.sh
```

### Text Classification - Taxi1500

First go to ``evaluation/taxi1500`` and run:

```
bash evaluate_all.sh
```

### Named Entity Recognition

For NER, first go to ``evaluation/tagging`` and run:
```
bash evaluate_all_ner.sh
```

### Part-Of-Speech Tagging

For POS, first go to ``evaluation/tagging`` and run:
```
bash evaluate_all_pos.sh
```

## Citation

If you find our code, models, or data useful for your research, please considering citing:

```
@article{liu2024translico,
  title={TransliCo: A Contrastive Learning Framework to Address the Script Barrier in Multilingual Pretrained Language Models},
  author={Yihong Liu and Chunlan Ma and Haotian Ye and Hinrich Sch{\"u}tze},
  journal={arXiv preprint arXiv:2401.06620},
  year={2024}
}
```

or

```
@inproceedings{imanigooghari-etal-2023-glot500,
	title        = {Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages},
	author       = {ImaniGooghari, Ayyoob  and Lin, Peiqin  and Kargaran, Amir Hossein  and Severini, Silvia  and Jalili Sabet, Masoud  and Kassner, Nora  and Ma, Chunlan  and Schmid, Helmut  and Martins, Andr{\'e}  and Yvon, Fran{\c{c}}ois  and Sch{\"u}tze, Hinrich},
	year         = 2023,
	month        = jul,
	booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
	publisher    = {Association for Computational Linguistics},
	address      = {Toronto, Canada},
	pages        = {1082--1117},
	url          = {https://aclanthology.org/2023.acl-long.61}
}
```

```
@article{liu2023ofa,
  title={OFA: A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining},
  author={Liu, Yihong and Lin, Peiqin and Wang, Mingyang and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2311.08849},
  year={2023}
}
```

## Acknowledgements

This repository is built on top of [xtreme](https://github.com/google-research/xtreme), [Glot500](https://github.com/cisnlp/Glot500) and [OFA](https://github.com/cisnlp/ofa).
