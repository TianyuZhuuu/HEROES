import argparse
import json
import os


def clean_abstract_text(abstract_text):
    reference = []
    for s in abstract_text:
        reference.append(' '.join(s.split()[1:-1]))
    return reference


def main(args):
    dataset_dir = args.dataset_dir

    for corpus_type in ['train', 'val', 'test']:
        output_dir = os.path.join(dataset_dir, 'input', 'full', corpus_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        doc_skip = 0
        sect_skip = 0
        sent_skip = 0

        filepath = os.path.join(dataset_dir, f'{corpus_type}.txt')
        with open(filepath) as f:
            for line in f:
                input_data = json.loads(line)

                id = input_data['article_id']
                reference = clean_abstract_text(input_data['abstract_text'])
                article_text = input_data['article_text']
                titles = input_data['section_names']
                sections = input_data['sections']

                assert len(titles) == len(sections)

                # empty document
                if len((' '.join(article_text)).split()) == 0:
                    doc_skip += 1
                    continue

                valid_titles, valid_sections = [], []
                for title, section in zip(titles, sections):
                    valid_sents = []
                    for sent in section:
                        words = sent.split()
                        # empty sentence
                        if len(words) == 0:
                            sent_skip += 1
                            continue
                        valid_sents.append(' '.join(words))

                    # empty section
                    if len(valid_sents) == 0:
                        sect_skip += 1
                        continue
                    valid_titles.append(title)
                    valid_sections.append(valid_sents)

                if len(valid_titles) == 0:
                    doc_skip += 1
                    continue

                assert all([len(section) != 0 for section in valid_sections])

                output_data = {
                    'id': id,
                    'reference': reference,
                    'titles': valid_titles,
                    'sections': valid_sections
                }
                output_path = os.path.join(output_dir, f'{id}.json')
                with open(output_path, 'w') as wf:
                    json.dump(output_data, wf)

        print(
            f'Skip {doc_skip} documents, {sect_skip} sections, {sent_skip} sentences in the {corpus_type} set')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    args = parser.parse_args()

    main(args)