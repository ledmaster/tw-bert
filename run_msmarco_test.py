import polars as pl
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader

import tw_bert_v2

from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from joblib import Parallel, delayed
from datasets import load_dataset

stemmer = PorterStemmer()
english_stopwords = list(stopwords.words('english'))

def clean_and_count(pid, passage):
	stemmed_tokens = [stemmer.stem(w) for w in passage if w not in english_stopwords]
	token_counts = Counter(stemmed_tokens)
	return (pid, token_counts)

def clean_query(query_tokens):
	cleaned_tokens = [stemmer.stem(w) for w in query_tokens if w not in english_stopwords]
	return cleaned_tokens

def query_tf_vec(cleaned_tokens):
	token_counts = Counter(cleaned_tokens)
	tf_vector = [token_counts[w] for w in cleaned_tokens]
	return tf_vector

class MSMARCOData(Dataset):
	def __init__(self, queries):
		self.queries = queries
		
	def __len__(self):
		return len(self.queries)
	
	def __getitem__(self, index):
		qid = self.queries[index]
		query = qid_map[qid]
		query_tf_vec = query_tf_vecs_map[qid]
		corpus = [document_word_frequencies[d] for d in query_doc_map[qid]]
		targets = query_labels_map[qid]
		return query, query_tf_vec, corpus, targets
	



if __name__ == "__main__":
	torch.set_default_device('cuda')
	
	# Load MSMARCO dataset using datasets library
	msmarco_dataset = load_dataset("microsoft/ms_marco", "v1.1")
	
	# Process the dataset - each row contains a query with multiple passages
	# Select only 1000 samples for training and validation
	processed_rows = []
	
	for i, row in enumerate(msmarco_dataset['train']):
		if i >= 1000:
			break
		qid = row['query_id']
		query = row['query']
		passages = row['passages']
		
		# Extract passage texts and labels
		passage_texts = passages['passage_text']
		is_selected = passages['is_selected']
		
		# Create individual query-passage pairs
		for i, (passage, label) in enumerate(zip(passage_texts, is_selected)):
			processed_rows.append({
				'qid': qid,
				'pid': f"{qid}_{i}",  # Create unique passage ID
				'query': query,
				'passage': passage,
				'label': label
			})
	
	# Convert to polars DataFrame
	data = pl.DataFrame(processed_rows)
	
	# Group by query and take top 100 passages per query
	data = data.sort('label', descending=True).group_by('qid').head(100)
	queries = data.select(pl.col("qid")).unique().to_numpy().squeeze()
	train_queries = queries[:-500]
	val_queries = queries[-500:]

	query_tf_vecs = data.select(pl.col("qid"), pl.col('query')).unique().with_columns([
		pl.col("query").str.extract_all(r"[A-Za-z0-9']+").alias("query_terms")]).with_columns([
			pl.col("query_terms").map_elements(clean_query, return_dtype=pl.List(pl.String)).alias("clean_query_terms")]).with_columns([
				pl.col("clean_query_terms").map_elements(query_tf_vec, return_dtype=pl.List(pl.Int32)).alias("query_tf_vec"),
				pl.col("clean_query_terms").list.join(" ").alias("clean_query")])
			
	
	doc_lists = data.select(pl.col("pid"), pl.col('passage')).unique().with_columns([
		pl.col("passage").str.to_lowercase().str.extract_all(r"[A-Za-z0-9']+").alias("passage_terms")]).select(pl.col("pid", "passage_terms"))

	# parallelize the loop using joblib
	document_word_frequencies = dict(Parallel(n_jobs=-1)(delayed(clean_and_count)(doc[0], doc[1]) for doc in doc_lists.rows()))
			
	query_doc_map = {row[0]: row[1] for row in data.select(pl.col("qid"), pl.col("pid")).group_by("qid").agg(pl.col("pid")).rows()}
	query_labels_map = {row[0]: row[1] for row in data.select(pl.col("qid"), pl.col("label")).group_by("qid").agg(pl.col("label")).rows()}
	qid_map = {row[0]: row[1] for row in query_tf_vecs.select(pl.col("qid"), pl.col("clean_query")).unique().rows()}
	query_tf_vecs_map = {row[0]: row[1] for row in query_tf_vecs.select(pl.col("qid"), pl.col("query_tf_vec")).rows()}
 
	train_dataset = MSMARCOData(train_queries)
	val_dataset = MSMARCOData(val_queries)
 
	model = tw_bert_v2.TWBERT().cuda()
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
	criterion = tw_bert_v2.TWBERTLossFT()
 
	gradient_accumulation_steps = 500
	global_step = 0
	for epoch in range(10):
		for batch_idx in range(len(train_dataset)):
			global_step += 1
			query, query_tf_vec, corpus, target = train_dataset[batch_idx]
			query_tf_vec = torch.tensor(query_tf_vec, dtype=torch.float32, device="cuda")
			avg_doc_len = 500
			if sum(target) == 0 or len(corpus) == 0:
				continue
			tokenized_query, mask = tw_bert_v2.token_and_mask_query(query, tw_bert_v2.tokenizer)
			term_weights = model(tokenized_query["input_ids"], tokenized_query["attention_mask"], mask).squeeze()[1:-1]
			output = tw_bert_v2.score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len)
			target = torch.tensor(target, dtype=torch.float32, device="cuda")
			loss = criterion(output, target)
			loss = loss / gradient_accumulation_steps
			
			loss.backward()
			
			if (global_step % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataset)):
				print("Step: ", global_step, "Train Loss: ", loss.item() * gradient_accumulation_steps)
				optimizer.step()
				optimizer.zero_grad()
				
				with torch.no_grad():
					model.eval()
					validation_loss = 0
					mrr = 0
					for val_idx in range(len(val_dataset)):
						query, query_tf_vec, corpus, target = val_dataset[val_idx]
						query_tf_vec = torch.tensor(query_tf_vec, dtype=torch.float32, device="cuda")
						avg_doc_len = 500
						if sum(target) == 0 or len(corpus) == 0:
							continue
						tokenized_query, mask = tw_bert_v2.token_and_mask_query(query, tw_bert_v2.tokenizer)
						term_weights = model(tokenized_query["input_ids"], tokenized_query["attention_mask"], mask).squeeze()[1:-1]
						#term_weights = torch.ones(term_weights.shape, device="cuda") # check non-optimized metrics
						output = tw_bert_v2.score_vec(query, query_tf_vec, corpus, term_weights, avg_doc_len)
						target = torch.tensor(target, dtype=torch.float32, device="cuda")
						
						if term_weights.sum() != 0:
							mrr += 1 / (torch.nonzero(target[output.sort(descending=True).indices])[0].item()+1)
						else:
							mrr += 0.
						validation_loss += criterion(output, target).item()
					print("Val Loss: ", validation_loss / len(val_dataset), "Val MRR: ", mrr / len(val_dataset))
					model.train()