import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.module.embedding import WordEmbedding
from common.module.vocabulary import Vocab
from common.utils.general import load_args, seed_everything
from content_ranking.module.dataloader import ContentRankingDataset, BucketSampler, collate_fn
from content_ranking.module.model import ContentRankingModule
from content_ranking.module.training import scoring_step


def save_ranking_results(full_input_dir, ranking_dir, cache):
    skip = 0
    if not os.path.exists(ranking_dir):
        os.makedirs(ranking_dir)

    for filename in os.listdir(full_input_dir):
        if not filename.endswith('.json'):
            skip += 1
            continue

        input_path = os.path.join(full_input_dir, filename)
        output_path = os.path.join(ranking_dir, filename)
        with open(input_path) as f:
            input_data = json.load(f)
        id = input_data['id']
        sections = input_data['sections']

        sect_score = [0 for _ in range(len(sections))]
        sent_scores = [[0 for _ in section] for section in sections]

        sect_cache = cache[id]['sect_score']
        sent_cache = cache[id]['sent_scores']

        for sect_id in range(len(sections)):
            if sect_id in sect_cache:
                sect_score[sect_id] = sect_cache[sect_id]
                m = len(sections[sect_id])
                n = len(sent_cache[sect_id])
                for j in range(min(m, n)):
                    sent_scores[sect_id][j] = sent_cache[sect_id][j]

        output_data = {'sect_score': sect_score, 'sent_scores': sent_scores}
        with open(output_path, 'w') as f:
            json.dump(output_data, f)

    print(f'Skip {skip} instances')


def main(args):
    hps_path = os.path.join(args.experiment_dir, 'hps.json')
    ckpt_path = os.path.join(args.experiment_dir, 'checkpoint.pt')
    val_ranking_dir = os.path.join(args.experiment_dir, 'ranking', 'val')
    test_ranking_dir = os.path.join(args.experiment_dir, 'ranking', 'test')

    hps = load_args(hps_path)
    dataset_dir = hps.dataset_dir

    seed_everything(hps.seed)
    device = torch.device(hps.device)

    # Vocabulary
    vocab_path = os.path.join(dataset_dir, 'vocab')
    vocab = Vocab(vocab_path, hps.vocab_size)

    # Validation set
    val_input_dir = os.path.join(dataset_dir, 'input', 'content_ranking', 'val')
    val_full_input_dir = os.path.join(dataset_dir, 'input', 'full', 'val')
    val_label_dir = os.path.join(dataset_dir, 'label', 'content_ranking', 'val')
    val_dataset = ContentRankingDataset(vocab, hps.title_max_words, hps.max_words, hps.max_sents, val_input_dir,
                                        val_label_dir)
    val_batch_per_bucket = (len(val_dataset) + hps.batch_size - 1) // hps.batch_size
    val_sampler = BucketSampler(val_dataset, val_batch_per_bucket * hps.batch_size, hps.batch_size)
    val_loader = DataLoader(val_dataset, hps.batch_size, sampler=val_sampler, collate_fn=collate_fn)

    # Test set
    test_input_dir = os.path.join(dataset_dir, 'input', 'content_ranking', 'test')
    test_full_input_dir = os.path.join(dataset_dir, 'input', 'full', 'test')
    test_label_dir = os.path.join(dataset_dir, 'label', 'content_ranking', 'test')
    test_dataset = ContentRankingDataset(vocab, hps.title_max_words, hps.max_words, hps.max_sents, test_input_dir,
                                         test_label_dir)
    test_batch_per_bucket = (len(test_dataset) + hps.batch_size - 1) // hps.batch_size
    test_sampler = BucketSampler(test_dataset, test_batch_per_bucket * hps.batch_size, hps.batch_size)
    test_loader = DataLoader(test_dataset, hps.batch_size, sampler=test_sampler, collate_fn=collate_fn)

    # Embedding weights
    embedding = WordEmbedding(hps.embedding_path, vocab).load_word_vctrs()

    # Model
    model = ContentRankingModule(embedding, hps.hidden_size, hps.title_pooling, hps.word_pooling,
                                 hps.sent_pooling, hps.aggregation_strategy)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_cache, test_cache = dict(), dict()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Inference Val ') as pbar:
            for step, data in enumerate(val_loader):
                scoring_step(data, model, val_cache, device)
                pbar.update(1)

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc=f'Inference Test') as pbar:
            for step, data in enumerate(test_loader):
                scoring_step(data, model, test_cache, device)
                pbar.update(1)

    save_ranking_results(val_full_input_dir, val_ranking_dir, val_cache)
    save_ranking_results(test_full_input_dir, test_ranking_dir, test_cache)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='experiments/arxiv/content_ranking')
    args = parser.parse_args()

    main(args)
