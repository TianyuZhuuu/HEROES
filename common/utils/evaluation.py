from multiprocessing import Pool

import numpy as np
from rouge import rouge


def rouge_scores(hyp_ref_pair):
    hypothesis, reference = hyp_ref_pair
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=False,
                            alpha=0.5,  # Default F1_score
                            stemming=True)
    scores = evaluator.get_scores(hypothesis, reference)
    return scores


def rouge_mean(hypothesis, reference):
    scores = rouge_scores((hypothesis, reference))
    rouge_1 = scores['rouge-1']['f']
    rouge_2 = scores['rouge-2']['f']
    rouge_l = scores['rouge-l']['f']
    return (rouge_1 + rouge_2 + rouge_l) / 3.0


def fast_evaluation(system_summaries, model_summaries):
    with Pool(12) as p:
        results = p.map(rouge_scores, zip(system_summaries, model_summaries))

    rouge_1_p = np.mean([scores['rouge-1']['p'] for scores in results])
    rouge_1_r = np.mean([scores['rouge-1']['r'] for scores in results])
    rouge_1_f = np.mean([scores['rouge-1']['f'] for scores in results])

    rouge_2_p = np.mean([scores['rouge-2']['p'] for scores in results])
    rouge_2_r = np.mean([scores['rouge-2']['r'] for scores in results])
    rouge_2_f = np.mean([scores['rouge-2']['f'] for scores in results])

    rouge_l_p = np.mean([scores['rouge-l']['p'] for scores in results])
    rouge_l_r = np.mean([scores['rouge-l']['r'] for scores in results])
    rouge_l_f = np.mean([scores['rouge-l']['f'] for scores in results])

    scores = dict()
    scores['rouge-1'] = {'p': rouge_1_p, 'r': rouge_1_r, 'f': rouge_1_f}
    scores['rouge-2'] = {'p': rouge_2_p, 'r': rouge_2_r, 'f': rouge_2_f}
    scores['rouge-l'] = {'p': rouge_l_p, 'r': rouge_l_r, 'f': rouge_l_f}

    return scores


def rouge2string(scores):
    res = f"""ROUGE-1-p: {scores['rouge-1']['p']:.6f} ROUGE-1-r: {scores['rouge-1']['r']:.6f} ROUGE-1-f: {scores['rouge-1']['f']:.6f}
ROUGE-2-p: {scores['rouge-2']['p']:.6f} ROUGE-2-r: {scores['rouge-2']['r']:.6f} ROUGE-2-f: {scores['rouge-2']['f']:.6f}
ROUGE-l-p: {scores['rouge-l']['p']:.6f} ROUGE-l-r: {scores['rouge-l']['r']:.6f} ROUGE-l-f: {scores['rouge-l']['f']:.6f}"""
    print(res)
    return res
