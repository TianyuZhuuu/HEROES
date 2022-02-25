import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from common.utils.evaluation import rouge_mean, rouge_scores


def prepare_data(idx, filenames, full_input_dir, ranking_input_dir, ranking_label_dir):
    full_input_path = os.path.join(full_input_dir, filenames[idx])

    with open(full_input_path) as f:
        input_data = json.load(f)

    id = input_data['id']
    reference = input_data['reference']
    reference_str = '\n'.join(reference)
    titles = input_data['titles']
    sections = input_data['sections']

    for sect_idx, (title, section) in enumerate(zip(titles, sections)):
        sent_rouges = [rouge_mean(sent, reference_str) for sent in section]
        sect_str = '\n'.join(section)
        sect_rouge = rouge_scores((sect_str, reference_str))['rouge-2']['r']

        rank_inst_id = f'{id}_{sect_idx}'

        ranking_input_path = os.path.join(ranking_input_dir, f'{rank_inst_id}.json')
        ranking_input_data = {'id': rank_inst_id, 'title': title, 'section': section}
        with open(ranking_input_path, 'w') as f:
            json.dump(ranking_input_data, f)

        ranking_label_path = os.path.join(ranking_label_dir, f'{rank_inst_id}.json')
        ranking_label_data = {'id': rank_inst_id, 'sect_rouge': sect_rouge, 'sent_rouges': sent_rouges}
        with open(ranking_label_path, 'w') as f:
            json.dump(ranking_label_data, f)


def helper(full_input_dir, ranking_input_dir, ranking_label_dir):
    if not os.path.exists(ranking_input_dir):
        os.makedirs(ranking_input_dir)
    if not os.path.exists(ranking_label_dir):
        os.makedirs(ranking_label_dir)
    filenames = sorted(os.listdir(full_input_dir))
    N = len(filenames)
    process_fn = partial(prepare_data, filenames=filenames, full_input_dir=full_input_dir,
                         ranking_input_dir=ranking_input_dir, ranking_label_dir=ranking_label_dir)
    with Pool(16) as p:
        list(tqdm(p.imap(process_fn, range(N)), total=N))


def main(args):
    for corpus_type in ['train', 'val', 'test']:
        full_input_dir = os.path.join(args.dataset_dir, 'input', 'full', corpus_type)
        ranking_input_dir = os.path.join(args.dataset_dir, 'input', 'content_ranking', corpus_type)
        ranking_label_dir = os.path.join(args.dataset_dir, 'label', 'content_ranking', corpus_type)
        helper(full_input_dir, ranking_input_dir, ranking_label_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    args = parser.parse_args()

    main(args)
