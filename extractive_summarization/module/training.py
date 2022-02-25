import torch

from common.utils.functions import masked_mean


def train_step(data, model, optimizer, max_grad_norm, device):
    graph = data['graph'].to(device)
    input_ids = data['input_ids'].to(device)
    labels = data['labels'].to(device)

    optimizer.zero_grad()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    scores = model(graph, input_ids)
    losses = loss_fn(scores, labels)
    sent_mask = input_ids.ne(0).any(dim=-1)
    loss = masked_mean(losses.view(-1), sent_mask.view(-1), dim=-1)
    loss_val = loss.item()

    loss.backward()
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss_val


def eval_step(data, model, word_limit, sent_limit, device):
    graph = data['graph'].to(device, non_blocking=True)
    input_ids = data['input_ids'].to(device, non_blocking=True)
    labels = data['labels'].to(device, non_blocking=True)
    batch_sections = data['sections']

    batch_summary = []
    batch_reference = ['\n'.join(reference) for reference in data['reference']]

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    scores = model(graph, input_ids)
    losses = loss_fn(scores, labels)
    sent_mask = input_ids.ne(0).any(dim=-1)
    loss = masked_mean(losses.view(-1), sent_mask.view(-1), dim=-1)

    batch_scores = torch.sigmoid(model(graph, input_ids))  # [batch x max_sects x max_sents]
    sent_mask = input_ids.ne(0).any(dim=-1)  # [batch x max_sects x max_sents]

    batch, max_sects, max_sents = batch_scores.shape

    for i in range(len(batch_sections)):
        sections = batch_sections[i]

        flat_scores = batch_scores[i].reshape(-1)
        flat_mask = sent_mask[i].reshape(-1)

        length = flat_scores.shape[-1]
        sorted_sids = sorted(list(range(length)), key=lambda idx: -flat_scores[idx].item())

        wc, sc = 0, 0
        selected_sent_indices = []
        for sid in sorted_sids:
            sect_idx, sent_idx = sid // max_sents, sid % max_sents

            if not flat_mask[sid].item():
                continue

            sent = sections[sect_idx][sent_idx]
            selected_sent_indices.append((sect_idx, sent_idx))
            count = len(sent.split())

            wc += count
            if word_limit != -1 and wc >= word_limit:
                break

            sc += 1
            if sent_limit != -1 and sc >= sent_limit:
                break

        selected_sent_indices.sort()

        summary = [sections[tup[0]][tup[1]] for tup in selected_sent_indices]
        batch_summary.append('\n'.join(summary))

    return loss.item(), batch_summary, batch_reference
