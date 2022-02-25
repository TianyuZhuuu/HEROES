import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.module.embedding import WordEmbedding
from common.module.vocabulary import Vocab
from common.utils.general import seed_everything, dump_args
from content_ranking.module.dataloader import ContentRankingDataset, BucketSampler, collate_fn
from content_ranking.module.model import ContentRankingModule
from content_ranking.module.training import train_step, eval_step


def main(args):
    print(args)

    seed_everything(args.seed)
    device = torch.device(args.device)

    dataset_dir = args.dataset_dir
    dataset = os.path.split(dataset_dir)[1].split('-')[0]
    experiment_dir = os.path.join(args.base_experiments_dir, dataset, 'content_ranking')
    os.makedirs(experiment_dir, exist_ok=True)

    # File save paths
    hps_path = os.path.join(experiment_dir, 'hps.json')
    performance_path = os.path.join(experiment_dir, 'performance')
    ckpt_path = os.path.join(experiment_dir, 'checkpoint.pt')

    # Vocabulary
    vocab_path = os.path.join(dataset_dir, 'vocab')
    vocab = Vocab(vocab_path, args.vocab_size)

    # Training set
    trn_input_dir = os.path.join(dataset_dir, 'input', 'content_ranking', 'train')
    trn_label_dir = os.path.join(dataset_dir, 'label', 'content_ranking', 'train')
    trn_dataset = ContentRankingDataset(vocab, args.title_max_words, args.max_words, args.max_sents, trn_input_dir,
                                        trn_label_dir)
    trn_sampler = BucketSampler(trn_dataset, args.batch_per_bucket * args.batch_size, args.batch_size)
    trn_loader = DataLoader(trn_dataset, args.batch_size, sampler=trn_sampler, collate_fn=collate_fn, num_workers=4,
                            pin_memory=True)

    # Validation set
    val_input_dir = os.path.join(dataset_dir, 'input', 'content_ranking', 'val')
    val_label_dir = os.path.join(dataset_dir, 'label', 'content_ranking', 'val')
    val_dataset = ContentRankingDataset(vocab, args.title_max_words, args.max_words, args.max_sents, val_input_dir,
                                        val_label_dir)
    val_batch_per_bucket = (len(val_dataset) + args.batch_size - 1) // args.batch_size
    val_sampler = BucketSampler(val_dataset, val_batch_per_bucket * args.batch_size, args.batch_size)
    val_loader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler, collate_fn=collate_fn)

    # Test set
    test_input_dir = os.path.join(dataset_dir, 'input', 'content_ranking', 'test')
    test_label_dir = os.path.join(dataset_dir, 'label', 'content_ranking', 'test')
    test_dataset = ContentRankingDataset(vocab, args.title_max_words, args.max_words, args.max_sents, test_input_dir,
                                         test_label_dir)
    test_batch_per_bucket = (len(test_dataset) + args.batch_size - 1) // args.batch_size
    test_sampler = BucketSampler(test_dataset, test_batch_per_bucket * args.batch_size, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, collate_fn=collate_fn)

    # Embedding weights
    embedding = WordEmbedding(args.embedding_path, vocab).load_word_vctrs()

    # Model
    model = ContentRankingModule(embedding, args.hidden_size, args.title_pooling, args.word_pooling,
                                 args.sent_pooling, args.aggregation_strategy)
    model.to(device)

    # Model training misc
    optimizer = Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)

    best_val_loss = 1e10

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        with tqdm(total=len(trn_loader), desc=f'Epoch {epoch: <2} Train') as pbar:
            for step, data in enumerate(trn_loader):
                train_step(data, model, optimizer, args.max_grad_norm, device)
                pbar.update(1)

        model.eval()
        total_sect, total_sent = 0, 0
        val_sect_loss, val_sent_loss = 0.0, 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Epoch {epoch: <2} Val') as pbar:
                for step, data in enumerate(val_loader):
                    sect_count, sent_count, sect_loss, sent_loss = eval_step(data, model, device)
                    total_sect += sect_count
                    total_sent += sent_count
                    val_sect_loss += sect_loss
                    val_sent_loss += sent_loss
                    pbar.update(1)

        val_sect_loss /= total_sect
        val_sent_loss /= total_sent
        val_total_loss = val_sect_loss + val_sent_loss

        print(f'Epoch {epoch}:')
        print('Val  losses:')
        print(f'    total: {val_total_loss:.6f} sect: {val_sect_loss:.6f} sent: {val_sent_loss:.6f}')

        total_sect, total_sent = 0, 0
        test_sect_loss, test_sent_loss = 0.0, 0.0

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f'Inference Test') as pbar:
                for step, data in enumerate(test_loader):
                    sect_count, sent_count, sect_loss, sent_loss = eval_step(data, model, device)
                    total_sect += sect_count
                    total_sent += sent_count
                    test_sect_loss += sect_loss
                    test_sent_loss += sent_loss
                    pbar.update(1)

        test_sect_loss /= total_sect
        test_sent_loss /= total_sent
        test_total_loss = test_sect_loss + test_sent_loss

        print('Test losses:')
        print(f'    total: {test_total_loss:.6f} sect: {test_sect_loss:.6f} sent: {test_sent_loss:.6f}')

        if val_total_loss < best_val_loss:
            dump_args(args, hps_path)
            print(f'Hyper parameters saved in {hps_path}')

            best_val_loss = val_total_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f'Checkpoint saved in {ckpt_path}')

            with open(performance_path, 'w', encoding='utf-8') as f:
                f.write('Val  losses:\n')
                f.write(f'    total: {val_total_loss:.6f} sect: {val_sect_loss:.6f} sent: {val_sent_loss:.6f}\n')
                f.write('Test losses:\n')
                f.write(f'    total: {test_total_loss:.6f} sect: {test_sect_loss:.6f} sent: {test_sent_loss:.6f}\n')
            print(f'Model performance saved in {performance_path}')

    print('Done training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Directory
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    parser.add_argument('--embedding_path', type=str, default='data/embeddings/glove.42B.300d.txt')
    parser.add_argument('--base_experiments_dir', type=str, default='experiments')

    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--title_max_words', type=int, default=10)
    parser.add_argument('--max_words', type=int, default=30)
    parser.add_argument('--max_sents', type=int, default=150)

    # Architecture
    parser.add_argument('--title_pooling', type=str, default='attention')
    parser.add_argument('--word_pooling', type=str, default='attention')
    parser.add_argument('--sent_pooling', type=str, default='attention')
    parser.add_argument('--aggregation_strategy', type=str, default='attention')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--batch_per_bucket', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    main(args)
