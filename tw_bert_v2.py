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
        self.scoring_layer = torch.nn.Linear(768, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, input_ids, attention_mask, term_mask):
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state # 1 x |W| x d
        expanded_term_mask = term_mask.unsqueeze(-1) # |T| x |W| x 1

        masked_embeddings = bert_output * expanded_term_mask # |T| x |W| x d - Masking
        pooled_embeddings = masked_embeddings.mean(dim=1)  # |T| x d - Pooling
        return self.relu(self.scoring_layer(pooled_embeddings)) # |T|

# listMLE adapted from https://github.com/allegro/allRank/tree/master/allrank/models/losses
class TWBERTLossFT(torch.nn.Module):
    def __init__(self, threshold_low=0.2, threshold_high=1):
        super().__init__()
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
    
    def forward(self, scores, labels):
        absolute_errors = torch.abs(scores - labels)
        amse_loss = torch.zeros(scores.shape[0])
        amse_loss = torch.where((absolute_errors >= self.threshold_low) & (absolute_errors < self.threshold_high), 
                        0.5 * (scores - labels)**2, amse_loss)
        amse_loss = torch.where(absolute_errors >= self.threshold_high, self.threshold_high * (absolute_errors - 0.5 * self.threshold_high), amse_loss)
        
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
    tokenized_query = tokenizer(query, return_tensors="pt", padding=True)
    ngrams = re.findall(r"[a-z0-9']+", query)
    tokenized_terms = [tokenizer(ng, add_special_tokens=False).input_ids for ng in ngrams]

    mask = torch.zeros(len(ngrams)+2, tokenized_query["input_ids"].shape[1])
    for i in range(1,len(ngrams)+1):
        for j in tokenized_terms[i-1]:
            mask[i, tokenized_query["input_ids"][0] == j] = 1
    return tokenized_query, mask

def score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len, k1=1.2, k3=8., b=0.75):
    # corpus = list of documents in word frequency format [{term: freq, ...}, {...}]
    query = re.findall(r"[a-z0-9']+", query)
    
    # Ensure query_tf_vec is on the same device as term_weights
    if isinstance(query_tf_vec, list):
        query_tf_vec = torch.tensor(query_tf_vec, device=term_weights.device, dtype=torch.float32)
    else:
        query_tf_vec = query_tf_vec.to(term_weights.device)
    
    weighted_query_terms = term_weights * query_tf_vec
    num_docs = len(corpus)
    
    query_idf = {}
    for term in query:
        #print(corpus)
        document_frequency = sum([1 for doc_tf in corpus if term in doc_tf])
        query_idf[term] = math.log((num_docs - document_frequency + 0.5)/(document_frequency+0.5) + 1) # +1?
    
    # Vectorized implementation: create document-term matrix
    num_terms = len(query)
    
    # Create document-term matrix and document lengths in one pass
    doc_term_matrix = torch.zeros(num_docs, num_terms, device=term_weights.device, dtype=torch.float32)
    doc_lengths = torch.zeros(num_docs, device=term_weights.device, dtype=torch.float32)
    
    for i, doc_tf in enumerate(corpus):
        doc_lengths[i] = sum(doc_tf.values())
        for j, term in enumerate(query):
            doc_term_matrix[i, j] = doc_tf.get(term, 0)
    
    # Vectorized BM25 calculations
    idf = torch.tensor([query_idf[term] for term in query], device=term_weights.device, dtype=torch.float32)
    
    # Broadcast operations across all documents
    numerator = doc_term_matrix * (k3 + 1) * weighted_query_terms.unsqueeze(0)  # [num_docs, num_terms]
    normalization_factor = k1 * ((1-b) + b * doc_lengths.unsqueeze(1) / avg_doc_len) + doc_term_matrix  # [num_docs, num_terms]
    denominator = (k3 + weighted_query_terms.unsqueeze(0)) * normalization_factor  # [num_docs, num_terms]
    
    # Final scoring with IDF weighting
    doc_scores = torch.sum(idf.unsqueeze(0) * numerator / denominator, dim=1)  # [num_docs]
    
    return doc_scores
