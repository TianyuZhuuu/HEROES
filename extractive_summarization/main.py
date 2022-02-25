import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.module.embedding import WordEmbedding
from common.module.vocabulary import Vocab
from common.utils.evaluation import rouge2string, fast_evaluation
from common.utils.general import seed_everything, str2bool, dump_args
from extractive_summarization.module.dataloader import SummarizationDataset, collate_fn
from extractive_summarization.module.model import HEROES
from extractive_summarization.module.training import train_step, eval_step


def main(args):
    print(args)

    dataset_dir = args.dataset_dir
    dataset = os.path.split(dataset_dir)[1].split('-')[0]
    experiment_dir = os.path.join(args.base_experiments_dir, dataset, args.setting, args.input_strategy,
                                  args.model_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    hps_path = os.path.join(experiment_dir, 'hps')
    rouge_path = os.path.join(experiment_dir, 'rouge')
    ckpt_path = os.path.join(experiment_dir, 'checkpoint.pt')

    seed_everything(args.seed)
    device = torch.device(args.device)

    # Vocabulary
    vocab_path = os.path.join(dataset_dir, 'vocab')
    vocab = Vocab(vocab_path, args.vocab_size)

    tfidf_path = os.path.join(dataset_dir, 'input', 'full', 'tfidf_ordered_words')

    # Training set
    trn_input_dir = os.path.join(dataset_dir, 'input', args.setting, args.input_strategy, 'train')
    trn_label_dir = os.path.join(dataset_dir, 'label', args.setting, args.input_strategy, 'train')
    trn_dataset = SummarizationDataset(vocab,
                                       args.title_max_num_words, args.max_num_sects, args.max_num_sents,
                                       args.max_num_words, args.max_dist,
                                       trn_input_dir, tfidf_path, trn_label_dir,
                                       args.use_sect_nodes, args.use_local_sent_conns, args.use_global_sent_conns)
    trn_loader = DataLoader(trn_dataset, args.batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)

    # Val set
    val_input_dir = os.path.join(dataset_dir, 'input', args.setting, args.input_strategy, 'val')
    val_label_dir = os.path.join(dataset_dir, 'label', args.setting, args.input_strategy, 'val')
    val_dataset = SummarizationDataset(vocab,
                                       args.title_max_num_words, args.max_num_sects, args.max_num_sents,
                                       args.max_num_words, args.max_dist,
                                       val_input_dir, tfidf_path, val_label_dir,
                                       args.use_sect_nodes, args.use_local_sent_conns, args.use_global_sent_conns)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    # Test set
    test_input_dir = os.path.join(dataset_dir, 'input', args.setting, args.input_strategy, 'test')
    test_label_dir = os.path.join(dataset_dir, 'label', args.setting, args.input_strategy, 'test')
    test_dataset = SummarizationDataset(vocab,
                                        args.title_max_num_words, args.max_num_sects, args.max_num_sents,
                                        args.max_num_words, args.max_dist,
                                        test_input_dir, tfidf_path, test_label_dir,
                                        args.use_sect_nodes, args.use_local_sent_conns, args.use_global_sent_conns)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    # Embedding weights
    embedding = WordEmbedding(args.embedding_path, vocab).load_word_vctrs()
    model = HEROES(args, embedding).to(device)

    optimizer = Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)

    best_rouge_mean = -1

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0.0
        with tqdm(total=len(trn_loader), desc=f'Epoch {epoch: <2} Train') as pbar:
            for step, data in enumerate(trn_loader):
                train_loss += train_step(data, model, optimizer, args.max_grad_norm, device) / len(trn_loader)
                pbar.update(1)
        print(f'Train loss: {train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        val_summaries, val_references = [], []
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f'Epoch {epoch: <2} Val') as pbar:
                for step, data in enumerate(val_loader):
                    loss, summaries, references = eval_step(data, model, args.word_limit, args.sent_limit, device)
                    val_loss += loss / len(val_loader)
                    val_summaries.extend(summaries)
                    val_references.extend(references)
                    pbar.update(1)
        val_scores = fast_evaluation(val_summaries, val_references)
        rouge_1, rouge_2, rouge_L = val_scores['rouge-1']['f'], val_scores['rouge-2']['f'], val_scores['rouge-l']['f']
        val_rouge_mean = (rouge_1 + rouge_2 + rouge_L) / 3.0
        print(f'Valid Loss: {val_loss:.4f} Rouge-mean: {val_rouge_mean:.6f}')
        val_rouge_string = rouge2string(val_scores)

        model.eval()
        test_loss = 0.0
        test_summaries, test_references = [], []
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f'Epoch {epoch} Test ') as pbar:
                for step, data in enumerate(test_loader):
                    loss, summaries, references = eval_step(data, model, args.word_limit, args.sent_limit, device)
                    test_loss += loss / len(test_loader)
                    test_summaries.extend(summaries)
                    test_references.extend(references)
                    pbar.update(1)
        test_scores = fast_evaluation(test_summaries, test_references)
        rouge_1, rouge_2, rouge_L = test_scores['rouge-1']['f'], test_scores['rouge-2']['f'], \
                                    test_scores['rouge-l']['f']
        test_rouge_mean = (rouge_1 + rouge_2 + rouge_L) / 3.0
        print(f'Test  Loss: {test_loss:.4f} Rouge-mean: {test_rouge_mean:.6f}')
        test_rouge_string = rouge2string(test_scores)

        if val_rouge_mean > best_rouge_mean:
            dump_args(args, hps_path)
            print(f'Hyper parameters saved in {hps_path}')

            with open(rouge_path, 'w', encoding='utf-8') as f:
                f.write(val_rouge_string + '\n')
                f.write('\n')
                f.write(test_rouge_string + '\n')

            best_rouge_mean = val_rouge_mean
            torch.save(model.state_dict(), ckpt_path)
            print(f'Checkpoint saved in {ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Directory
    parser.add_argument('--dataset_dir', default='data/arxiv-dataset')
    parser.add_argument('--setting', default='medium-resource')
    parser.add_argument('--input_strategy', default='ranking')
    parser.add_argument('--embedding_path', type=str, default='data/embeddings/glove.42B.300d.txt')
    parser.add_argument('--model_name', type=str, default='HEROES')
    parser.add_argument('--base_experiments_dir', type=str, default='experiments')
    parser.add_argument('--use_sect_nodes', type=str2bool, default='y')
    parser.add_argument('--use_local_sent_conns', type=str2bool, default='y')
    parser.add_argument('--use_global_sent_conns', type=str2bool, default='n')

    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--title_max_num_words', type=int, default=10)
    parser.add_argument('--max_num_sects', type=int, default=4)
    parser.add_argument('--max_num_sents', type=int, default=30)
    parser.add_argument('--max_num_words', type=int, default=50)
    parser.add_argument('--max_dist', type=int, default=50)

    parser.add_argument('--word_emb_dim', type=int, default=300)
    parser.add_argument('--sent_dim', type=int, default=512)
    parser.add_argument('--sect_dim', type=int, default=512)
    parser.add_argument('--word_num_heads', type=int, default=6)
    parser.add_argument('--sent_num_heads', type=int, default=8)
    parser.add_argument('--sect_num_heads', type=int, default=8)
    parser.add_argument('--ffn_inner_dim', type=int, default=2048)
    parser.add_argument('--dist_emb_dim', type=int, default=64)
    parser.add_argument('--lstm_dim', type=int, default=256)
    parser.add_argument('--n_feat_dim', type=int, default=128)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--num_iters', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument('--sent_limit', type=int, default=-1)
    parser.add_argument('--word_limit', type=int, default=200)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    main(args)
