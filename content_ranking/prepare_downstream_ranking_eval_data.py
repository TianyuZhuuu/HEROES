import argparse
import json
import os

from tqdm import tqdm


def prepare_downstream_ranking_data(input_dir, label_dir, content_ranking_dir, ranking_input_dir, ranking_label_dir, m,
                                    n):
    if not os.path.exists(ranking_input_dir):
        os.makedirs(ranking_input_dir)
    if not os.path.exists(ranking_label_dir):
        os.makedirs(ranking_label_dir)

    for name in tqdm(os.listdir(input_dir)):
        if not name.endswith('.json'):
            continue
        input_path = os.path.join(input_dir, name)
        label_path = os.path.join(label_dir, name)
        content_ranking_path = os.path.join(content_ranking_dir, name)
        ranking_input_path = os.path.join(ranking_input_dir, name)
        ranking_label_path = os.path.join(ranking_label_dir, name)

        with open(input_path) as f:
            input_data = json.load(f)
        with open(label_path) as f:
            label_data = json.load(f)
        with open(content_ranking_path) as f:
            ranking_data = json.load(f)

        id = input_data['id']
        reference = input_data['reference']
        titles = input_data['titles']
        sections = input_data['sections']
        labels = label_data['labels']
        sect_score = ranking_data['sect_score']
        sent_scores = ranking_data['sent_scores']

        top_sect_ids = list(range(len(sections)))
        top_sect_ids.sort(key=lambda id: -sect_score[id])
        top_sect_ids = top_sect_ids[:m]
        top_sect_ids.sort()

        ranking_titles = [titles[id] for id in top_sect_ids]
        ranking_sections, ranking_labels = [], []
        for idx in top_sect_ids:
            section = sections[idx]
            ranking_section = []
            ranking_label = []

            top_sent_ids = list(range(len(section)))
            top_sent_ids.sort(key=lambda sid: -sent_scores[idx][sid])
            top_sent_ids = top_sent_ids[:n]
            top_sent_ids.sort()

            for sid in top_sent_ids:
                sent_info = {'text': section[sid], 'pos': sid, 'start_dist': sid,
                             'end_dist': len(section) - sid - 1, 'rouge': sent_scores[idx][sid]}
                ranking_section.append(sent_info)
                ranking_label.append(labels[idx][sid])

            ranking_sections.append(ranking_section)
            ranking_labels.append(ranking_label)

        ranking_input_data = {'id': id, 'reference': reference, 'titles': ranking_titles, 'sections': ranking_sections}
        with open(ranking_input_path, 'w') as f:
            json.dump(ranking_input_data, f)

        ranking_label_data = {'id': id, 'labels': ranking_labels}
        with open(ranking_label_path, 'w') as f:
            json.dump(ranking_label_data, f)


def main(args):
    for corpus_type in ['val', 'test']:
        input_dir = os.path.join(args.dataset_dir, 'input', 'full', corpus_type)
        label_dir = os.path.join(args.dataset_dir, 'label', 'full', corpus_type)
        content_ranking_dir = os.path.join(args.ranking_dir, corpus_type)
        ranking_input_dir = os.path.join(args.dataset_dir, 'input', args.setting, 'ranking', corpus_type)
        ranking_label_dir = os.path.join(args.dataset_dir, 'label', args.setting, 'ranking', corpus_type)
        prepare_downstream_ranking_data(input_dir, label_dir, content_ranking_dir,
                                        ranking_input_dir, ranking_label_dir, args.m, args.n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    parser.add_argument('--setting', default='medium-resource')
    parser.add_argument('--ranking_dir', default='experiments/arxiv/content_ranking',
                        help='directory containing content ranking results')
    parser.add_argument('--m', default=4)
    parser.add_argument('--n', default=30)
    args = parser.parse_args()

    main(args)
