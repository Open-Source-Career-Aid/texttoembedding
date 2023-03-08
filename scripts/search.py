import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def open_file(filepath):
	with open(filepath, 'r', encoding='utf-8') as infile:
		return infile.read()

openai.api_key = None

def gpt3_embedding(content, engine='text-embedding-ada-002'):
	content = content.encode(encoding='ASCII',errors='ignore').decode()
	response = openai.Embedding.create(input=content,engine=engine)
	vector = response['data'][0]['embedding']  # this is a normal list
	return vector

def mean_pooling(model_output, attention_mask):
	    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
	    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mpnet_embedding(content):

	tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
	model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
	encoded_input = tokenizer(content, padding=True, truncation=True, return_tensors='pt')
	with torch.no_grad():
		model_output = model(**encoded_input)
	embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
	embedding = F.normalize(embeddings, p=2, dim=1)

	# converted to float, because tensor
	return [float(x) for x in embedding[0]]

def roberta_embedding(content):
	tokenizer = AutoTokenizer.from_pretrained('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt')
	model = AutoModel.from_pretrained('Maite89/Roberta_finetuning_semantic_similarity_stsb_multi_mt')
	encoded_input = tokenizer(content, padding=True, truncation=True, return_tensors='pt')
	with torch.no_grad():
		model_output = model(**encoded_input)
	embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
	embedding = F.normalize(embeddings, p=2, dim=1)
	return [float(x) for x in embedding[0]]

def similarity(v1, v2):  # return dot product of two vectors
	return np.dot(v1, v2)

def search_index(text, data, count=10, engine=''):
	if engine=='gpt3':
		vector = gpt3_embedding(text)
	elif engine=='mpnet':
		vector = mpnet_embedding(text)
	scores = list()
	for i in data:
		score = similarity(vector, i['vector'])
		#print(score)
		scores.append({'document': i['document'], 'heading': i['heading'], 'score': score})
	ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
	return ordered[0:count]

if __name__ == '__main__':

	while True:

		which_embedding = int(input("type 1 for ada, 2 for mpnet, 3 for roberta:"))

		if which_embedding==1:

			with open('index.json', 'r') as infile:
				data = json.load(infile)

		elif which_embedding==2:
			with open('index_mpnet.json', 'r') as infile:
				data = json.load(infile)

		elif which_embedding==3:
			with open('index_roberta.json', 'r') as infile:
				data = json.load(infile)

		query = input("Enter your question here: ")
		#print(query)
		if which_embedding==1:
			results = search_index(query, data, engine='gpt3')
		elif which_embedding==2:
			results = search_index(query, data, engine='mpnet')
		elif which_embedding==3:
			results = search_index(query, data, engine='roberta')
		[print(results[i]['document'], results[i]['heading'], results[i]['score']) for i in range(len(results))]