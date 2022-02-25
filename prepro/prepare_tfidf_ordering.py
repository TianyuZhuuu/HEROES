import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm


def main(args):
    dataset_dir = args.dataset_dir
    full_input_dir = os.path.join(dataset_dir, 'input', 'full', 'train')
    output_path = os.path.join(dataset_dir, 'input', 'full', 'tfidf_ordered_words')

    documents = []
    for path in tqdm(list(Path(full_input_dir).glob('*.json'))):
        with open(path) as f:
            input_data = json.load(f)
        document = ' '.join([sent for sect in input_data['sections'] for sent in sect])
        documents.append(document)

    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(documents)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)

    print(f'# of example: {len(documents)}, TFIDF vocabulary size: {len(vectorizer.vocabulary_)}')
    word_tfidf = np.array(tfidf_matrix.mean(0))
    del tfidf_matrix
    word_order = np.argsort(word_tfidf[0])

    id2word = vectorizer.get_feature_names()
    count = 0
    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx in word_order:
            w = id2word[idx]
            string = w + "\n"
            try:
                fout.write(string)
                count += 1
            except:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    args = parser.parse_args()

    main(args)
