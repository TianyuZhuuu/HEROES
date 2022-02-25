import torch.nn.utils
from torch.nn import BCEWithLogitsLoss

from common.utils.functions import masked_mean, masked_sum


def train_step(data, model, optimizer, max_grad_norm, device):
    title_input_ids = data['title_input_ids'].to(device, non_blocking=True)
    section_input_ids = data['section_input_ids'].to(device, non_blocking=True)
    sect_rouge = data['sect_rouge'].to(device, non_blocking=True)
    sent_rouges = data['sent_rouges'].to(device, non_blocking=True)

    loss_fn = BCEWithLogitsLoss(reduction='none')
    sect_score, sent_scores, extras = model(title_input_ids, section_input_ids)

    sect_loss = loss_fn(sect_score, sect_rouge).mean()
    sent_losses = loss_fn(sent_scores, sent_rouges)
    sent_mask = section_input_ids.ne(0).any(dim=-1)
    sent_loss = masked_mean(sent_losses.view(-1), sent_mask.view(-1), dim=-1)
    loss = sect_loss + sent_loss

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()


def eval_step(data, model, device):
    title_input_ids = data['title_input_ids'].to(device, non_blocking=True)
    section_input_ids = data['section_input_ids'].to(device, non_blocking=True)
    sect_rouge = data['sect_rouge'].to(device, non_blocking=True)
    sent_rouges = data['sent_rouges'].to(device, non_blocking=True)

    loss_fn = BCEWithLogitsLoss(reduction='none')
    sect_score, sent_scores, extras = model(title_input_ids, section_input_ids)

    sent_mask = section_input_ids.ne(0).any(-1)

    sect_count = sect_score.shape[0]
    sent_count = sent_mask.sum().item()

    sect_loss = loss_fn(sect_score, sect_rouge).sum().item()
    sent_losses = loss_fn(sent_scores, sent_rouges)
    sent_loss = masked_sum(sent_losses.view(-1), sent_mask.view(-1), dim=-1).item()

    return sect_count, sent_count, sect_loss, sent_loss


def scoring_step(data, model, cache, device):
    ids = data['id']
    title_input_ids = data['title_input_ids'].to(device, non_blocking=True)
    section_input_ids = data['section_input_ids'].to(device, non_blocking=True)

    sect_score, sent_scores, extras = model(title_input_ids, section_input_ids)
    sect_score = torch.sigmoid(sect_score)
    sent_scores = torch.sigmoid(sent_scores)

    for id, _sect_score, _sent_scores in zip(ids, sect_score, sent_scores):
        article_id, sect_id = id.rsplit('_', maxsplit=1)
        sect_id = int(sect_id)
        if article_id not in cache:
            cache[article_id] = dict()
            cache[article_id]['sect_score'] = dict()
            cache[article_id]['sent_scores'] = dict()
        cache[article_id]['sect_score'][sect_id] = _sect_score.item()
        cache[article_id]['sent_scores'][sect_id] = _sent_scores.cpu().numpy().tolist()
