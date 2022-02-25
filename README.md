# HEROES
Source code for CIKM 2021 paper: [Summarizing Long-Form Document with Rich Discourse Information](https://dl.acm.org/doi/abs/10.1145/3459637.3482396)

Part of the code is borrowed from [HSG](https://github.com/dqwang122/HeterSumGraph). Thanks the authors for making their code public.

## Dependencies
- python==3.8
- pytorch==1.7.1
- dgl-cuda11.0==0.6.1
- py-rouge==1.1

## Data Preparation
- Download the `arXiv` and `PubMed` dataset following the instructions [here](https://github.com/armancohan/long-summarization) and extract files under the `data` dir
- Download the `GloVe` embeddings [here](https://nlp.stanford.edu/projects/glove/) and extract files under the `data` dir

## Preprocessing

Execute python files under the _prepro_ folder in order for:
- Converting the original dataset in _jsonl_ format (one line per document) to _json_ format (one json file per document)
- Preparing data for content ranking. (input: a single section, label: ROUGE-recall for section and ROUGE-F1 for sentences)
- Preparing extraction labels for full document.
- Preparing training ranking-based digest for downstream summarization task.
- Preparing ordering of words based on tf-idf statistics (for word node filtering)

```
python prepro/save_in_json_format.py --dataset_dir data/arxiv-dataset
python prepro/prepare_content_ranking_data.py --dataset_dir data/arxiv-dataset
python prepro/prepare_content_ranking_data.py --dataset_dir data/arxiv-dataset --k 2
python prepro/prepare_downstream_ranking_train_data.py --dataset_dir data/arxiv-dataset --setting medium_resource --m 4 --n 30
python prepro/prepare_tfidf_ordering.py --dataset_dir data/arxiv-dataset
```

## Content Ranking

Execute the python files under the _content_ranking_ folder in order for:
- Performing content ranking (scoring sections and sentences)
- Writing the ranking results to files
- Preparing valid/test ranking-based digest through assembling top-scoring sections and sentences.

```
python content_ranking/main.py
python content_ranking/save_ranking_results.py
python content_ranking/prepare_downstream_ranking_eval_data.py
```

## Extractive Summarization

For execution, simply run

```
python extractive_summarization/main.py
```

> We have set the best performing hyperparameters as default, but feel free to experiment with different hyperparameters.

## Citation

If you are interest in our work, please cite as follows:
```
@inproceedings{zhu2021summarizing,
  title={Summarizing Long-Form Document with Rich Discourse Information},
  author={Zhu, Tianyu and Hua, Wen and Qu, Jianfeng and Zhou, Xiaofang},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={2770--2779},
  year={2021}
}
```
