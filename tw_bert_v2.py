from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
import torch
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
import math
import re
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
DEFAULT_EPS = 1e-8

class TWBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased").base_model
        self.linear = torch.nn.Linear(768, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, input_ids, attention_mask, term_mask):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state # 1 x |W| x d
        mask_ = term_mask.unsqueeze(-1) # |T| x |W| x 1

        q_h_masked = bert_output * mask_ # |T| x |W| x d - Masking
        #print(q_h_masked)
        p = q_h_masked.mean(dim=1)  # |T| x d - Pooling
        #print(p)
        return self.relu(self.linear(p)) # |T|

# listMLE adapted from https://github.com/allegro/allRank/tree/master/allrank/models/losses
class TWBERTLossFT(torch.nn.Module):
    def __init__(self, d1=0.2, d2=1):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
    
    def forward(self, scores, labels):
        a_scores = torch.abs(scores - labels)
        amse_loss = torch.zeros(scores.shape[0])
        amse_loss = torch.where((a_scores >= self.d1) & (a_scores < self.d2), 
                        0.5 * (scores - labels)**2, amse_loss)
        amse_loss = torch.where(a_scores >= self.d2, self.d2 * (a_scores - 0.5 * self.d2), amse_loss)
        
        random_indices = torch.randperm(scores.shape[-1])
        y_pred_shuffled = scores[random_indices]
        y_true_shuffled = labels[random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        preds_sorted_by_true = y_pred_shuffled[indices]

        max_pred_values, _ = preds_sorted_by_true.max(dim=0, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip([0]), 0).flip([0])

        observation_loss = torch.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

        return amse_loss.mean() + observation_loss.mean()
    
def token_and_mask_query(query, tokenizer):
    query_t = tokenizer(query, return_tensors="pt", padding=True)
    ngrams = re.findall(r"[a-z0-9']+", query)
    term_t = [tokenizer(ng, add_special_tokens=False).input_ids for ng in ngrams]

    mask = torch.zeros(len(ngrams)+2, query_t["input_ids"].shape[1])
    for i in range(1,len(ngrams)+1):
        for j in term_t[i-1]:
            mask[i, query_t["input_ids"][0] == j] = 1
    return query_t, mask

def score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len, k1=1.2, k3=8., b=0.75):
    # corpus = list of documents in word frequency format [{term: freq, ...}, {...}]
    query = re.findall(r"[a-z0-9']+", query)
    
    # Ensure query_tf_vec is on the same device as term_weights
    if isinstance(query_tf_vec, list):
        query_tf_vec = torch.tensor(query_tf_vec, device=term_weights.device, dtype=torch.float32)
    else:
        query_tf_vec = query_tf_vec.to(term_weights.device)
    
    f_ti_t_w = term_weights * query_tf_vec
    num_docs = len(corpus)
    
    query_idf = {}
    for term in query:
        #print(corpus)
        df_t = sum([1 for doc_tf in corpus if term in doc_tf])
        query_idf[term] = math.log((num_docs - df_t + 0.5)/(df_t+0.5) + 1) # +1?
    
    doc_scores = list()
    for doc_tf in corpus:
        doc_len = sum(doc_tf.values())
        # Ensure tensors are on the same device as term_weights
        doc_tf_vec = torch.tensor([doc_tf.get(term, 0) for term in query], device=term_weights.device, dtype=torch.float32)
        num = doc_tf_vec * (k3 + 1) * f_ti_t_w
        k = k1 * ((1-b) + b * doc_len/avg_doc_len) + doc_tf_vec
        den = (k3 + f_ti_t_w) * k
        idf = torch.tensor([query_idf[term] for term in query], device=term_weights.device, dtype=torch.float32)
        doc_scores.append(torch.sum(idf * num/den))
    
    
    return torch.stack(doc_scores)
