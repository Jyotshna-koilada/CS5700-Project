import numpy as np
import datasets
import transformers
import re
import torch
import tqdm
import random
import argparse
import functools
import time
import os
from graphs import save_roc_curves, save_ll_histograms, save_llr_histograms
from get_stats import get_ll, get_lls, pr_eval, roc_eval, get_rank, get_entropy
pattern = re.compile(r"<extra_id_\d+>")
def b_load():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    m_model.cpu()
    b_model.to(DEVICE)
def m_load():
    start = time.time()
    b_model.cpu()
    m_model.to(DEVICE)
def masker(text, sl, pct, ceil_pct=False):
    tok = text.split(' ')
    ms = '<<<mask>>>'
    spans = pct * len(tok) / (sl + 1 * 2)
    if ceil_pct:
        spans = np.ceil(spans)
    spans = int(spans)
    masks = 0
    while masks < spans:
        start = np.random.randint(0, len(tok) - sl)
        end = start + sl
        sest = max(0, start - 1)
        seend = min(len(tok), end + 1)
        if ms not in tok[sest:seend]:
            tok[start:end] = [ms]
            masks += 1
    filled = 0
    for idx, token in enumerate(tok):
        if token == ms:
            tok[idx] = f'<extra_id_{filled}>'
            filled += 1
    assert filled == masks, f"num_filled {filled} != n_masks {masks}"
    text = ' '.join(tok)
    return text
def m_count(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
def m_replace(texts):
    expected = m_count(texts)
    stid = m_tok.encode(f"<extra_id_{max(expected)}>")[0]
    tok = m_tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    final = m_model.generate(**tok, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1, eos_token_id=stid)
    return m_tok.batch_decode(final, skip_special_tokens=False)
def get_fills(texts):
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    got_fills = [pattern.split(x)[1:-1] for x in texts]
    got_fills = [[y.strip() for y in x] for x in got_fills]
    return got_fills
def app_mfill(masked_texts, got_fills):
    tokens = [x.split(' ') for x in masked_texts]
    expected = m_count(masked_texts)
    for idx, (text, fills, n) in enumerate(zip(tokens, got_fills, expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]
    texts = [" ".join(x) for x in tokens]
    return texts
def pert_cand(texts, sl, pct, ceil_pct=False):
    masked_texts = [masker(x, sl, pct, ceil_pct) for x in texts]
    raw_fills = m_replace(masked_texts)
    got_fills = get_fills(raw_fills)
    pert_texts = app_mfill(masked_texts, got_fills)
    attempts = 1
    while '' in pert_texts:
        idxs = [idx for idx, x in enumerate(pert_texts) if x == '']
        masked_texts = [masker(x, sl, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = m_replace(masked_texts)
        got_fills = get_fills(raw_fills)
        new_perturbed_texts = app_mfill(masked_texts, got_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            pert_texts[idx] = x
        attempts += 1
    return pert_texts
def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(pert_cand(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs
def generate_modeltext(texts, min_words=55, prompt_tokens=30):
    encoded = b_tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    encoded = {key: value[:, :prompt_tokens] for key, value in encoded.items()}
    decoded = ['' for _ in range(len(texts))]
    tries = 0
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        min_length = 150
        outputs = b_model.generate(**encoded, min_length=min_length, max_length=200, do_sample=True, pad_token_id=b_tok.eos_token_id, eos_token_id=b_tok.eos_token_id)
        decoded = b_tok.batch_decode(outputs, skip_special_tokens=True)
        tries += 1
    return decoded
def get_perturbation_results(span_length=10, n_perturbations=1):
    m_load()
    torch.manual_seed(0)
    np.random.seed(0)
    results = []
    ot = data["original"]
    st = data["sampled"]
    pert_crit = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)
    pst = pert_crit([x for x in st for _ in range(n_perturbations)])
    pot = pert_crit([x for x in ot for _ in range(n_perturbations)])
    assert len(pst) == len(st) * n_perturbations, f"Expected {len(st) * n_perturbations} perturbed samples, got {len(pst)}"
    assert len(pot) == len(ot) * n_perturbations, f"Expected {len(ot) * n_perturbations} perturbed samples, got {len(pot)}"
    for idx in range(len(ot)):
        results.append({
            "original": ot[idx],
            "sampled": st[idx],
            "perturbed_sampled": pst[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": pot[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    b_load()
    for plots in tqdm.tqdm(results, desc="Computing log likelihoods"):
        psll = get_lls(b_model, b_tok, plots["perturbed_sampled"])
        poll = get_lls(b_model, b_tok, plots["perturbed_original"])
        plots["original_ll"] = get_ll(b_model, b_tok, plots["original"])
        plots["sampled_ll"] = get_ll(b_model, b_tok, plots["sampled"])
        plots["all_perturbed_sampled_ll"] = psll
        plots["all_perturbed_original_ll"] = poll
        plots["perturbed_sampled_ll"] = np.mean(psll)
        plots["perturbed_original_ll"] = np.mean(poll)
        plots["perturbed_sampled_ll_std"] = np.std(psll) if len(psll) > 1 else 1
        plots["perturbed_original_ll_std"] = np.std(poll) if len(poll) > 1 else 1
    return results
def pert_exp(results, criterion, span_length=10, n_perturbations=1, n_samples=200):
    pres = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            pres['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            pres['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
            pres['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            pres['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])
    falposrt, trposrt, rocauc = roc_eval(pres['real'], pres['samples'])
    pr, re, pra = pr_eval(pres['real'], pres['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {rocauc}, PR AUC: {pra}")
    return {
        'name': name,
        'predictions': pres,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': rocauc,
            'fpr': falposrt,
            'tpr': trposrt,
        },
        'pr_metrics': {
            'pr_auc': pra,
            'precision': pr,
            'recall': re,
        },
        'loss': 1 - pra,
    }

def bt_exp(criterion_fn, name, n_samples=200):
    torch.manual_seed(0)
    np.random.seed(0)
    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]
        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": criterion_fn(b_model, b_tok, original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_crit": criterion_fn(b_model, b_tok, sampled_text[idx]),
            })
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }
    fpr, tpr, roc_auc = roc_eval(predictions['real'], predictions['samples'])
    p, r, pr_auc = pr_eval(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }
def strip_newlines(text):
    return ' '.join(text.split())
def trim_to_shorter_length(texta, textb):
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb
def truncate_to_substring(text, substring, idx_occurrence):
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]
def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }
    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = generate_modeltext(original_text, min_words=30 if args.dataset in ['pubmed'] else 55)
        for o, s in zip(original_text, sampled_text):
            o, s = trim_to_shorter_length(o, s)
            data["original"].append(o)
            data["sampled"].append(s)
    return data
def generate_data(dataset, key, n_samples = 200):
    data = datasets.load_dataset(dataset, split='train', cache_dir=cache_dir)[key]
    data = list(dict.fromkeys(data))
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data
    random.seed(0)
    random.shuffle(data)
    data = data[:5_000]
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    return generate_samples(data[:n_samples], batch_size=batch_size)
def sup_eval(data, model, n_samples = 200):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    real, fake = data['original'], data['sampled']
    with torch.no_grad():
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())
    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }
    fpr, tpr, roc_auc = roc_eval(real_preds, fake_preds)
    p, r, pr_auc = pr_eval(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    del detector
    torch.cuda.empty_cache()
    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--cache_dir', type=str, default="detect-gpt/cache")
    args = parser.parse_args()
    base_model_name = args.base_model_name
    folder = f"results/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    cache_dir = args.cache_dir
    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    print(f'Loading BASE model {args.base_model_name}...')
    b_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, cache_dir=cache_dir)
    optional_tok_kwargs = {}
    b_tok = transformers.AutoTokenizer.from_pretrained(base_model_name, **optional_tok_kwargs, cache_dir=cache_dir)
    b_tok.pad_token_id = b_tok.eos_token_id
    print(f'Loading mask filling model {mask_filling_model_name}...')
    m_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, cache_dir=cache_dir)
    try:
        n_positions = m_model.config.n_positions
    except AttributeError:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    m_tok = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    b_load()
    data = generate_data(args.dataset, args.dataset_key)
    b_outputs = [bt_exp(get_ll, "likelihood", n_samples=n_samples)]
    rank_criterion = lambda base_model, b_tok, text: -get_rank(base_model, b_tok, text, log=False)
    b_outputs.append(bt_exp(rank_criterion, "rank", n_samples=n_samples))
    logrank_criterion = lambda base_model, b_tok, text: -get_rank(base_model, b_tok, text, log=True)
    b_outputs.append(bt_exp(logrank_criterion, "log_rank", n_samples=n_samples))
    entropy_criterion = lambda base_model, b_tok, text: get_entropy(base_model, b_tok, text)
    b_outputs.append(bt_exp(entropy_criterion, "entropy", n_samples=n_samples))
    b_outputs.append(sup_eval(data, model='roberta-base-openai-detector'))
    b_outputs.append(sup_eval(data, model='roberta-large-openai-detector'))
    outputs = []
    for n_perturbations in n_perturbation_list:
        perturbation_results = get_perturbation_results(args.span_length, n_perturbations)
        for perturbation_mode in ['d', 'z']:
            output = pert_exp(perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
            outputs.append(output)
    outputs += b_outputs
    save_roc_curves(folder, outputs, base_model_name)
    save_ll_histograms(folder, outputs)
    save_llr_histograms(folder, outputs)