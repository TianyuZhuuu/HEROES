import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from common.utils.evaluation import rouge_mean


def prepare_extraction_label(idx, filenames, input_dir, label_dir):
    full_input_path = os.path.join(input_dir, filenames[idx])

    with open(full_input_path) as f:
        input_data = json.load(f)

    id = input_data['id']
    reference = input_data['reference']
    sections = input_data['sections']

    labels = [[0 for _ in section] for section in sections]

    for ref_sent in reference:
        tuples = []
        for i, section in enumerate(sections):
            for j, sent in enumerate(section):
                rouge = rouge_mean(sent, ref_sent)
                tuples.append((rouge, i, j))
        tuples.sort(key=lambda tup: -tup[0])
        if len(tuples) < args.k:
            print(id)
        for k in range(min(args.k, len(tuples))):
            labels[tuples[k][1]][tuples[k][2]] = 1

    output_data = {'id': id, 'labels': labels}
    output_path = os.path.join(label_dir, f'{id}.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f)


def helper(input_dir, label_dir):
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    filenames = sorted(os.listdir(input_dir))
    N = len(filenames)
    process_fn = partial(prepare_extraction_label, filenames=filenames, input_dir=input_dir, label_dir=label_dir)
    with Pool(16) as p:
        list(tqdm(p.imap(process_fn, range(N)), total=N))


def main(args):
    for corpus_type in ['train', 'val', 'test']:
        input_dir = os.path.join(args.dataset_dir, 'input', 'full', corpus_type)
        label_dir = os.path.join(args.dataset_dir, 'label', 'full', corpus_type)
        helper(input_dir, label_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    parser.add_argument('--k', default=2, help='the number of ground-truth extraction sentences per reference sentence')
    args = parser.parse_args()

    main(args)
