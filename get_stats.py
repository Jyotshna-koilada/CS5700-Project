import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch.nn.functional as F
DEVICE = DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_ll(base_model, base_tokenizer, text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()
def get_lls(base_model, base_tokenizer, texts):
    return [get_ll(base_model, base_tokenizer, text) for text in texts]
def get_rank(base_model, base_tokenizer, text, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
        ranks, timesteps = matches[:,-1], matches[:,-2]
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"
        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()
def get_entropy(base_model, base_tokenizer, text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()
def roc_eval(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)
def pr_eval(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)