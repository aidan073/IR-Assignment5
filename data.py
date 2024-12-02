import csv
import json
from rank_bm25 import BM25Okapi
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

def parseText(text)->str:
	return BeautifulSoup(text, "html.parser").text

def getTopics(topics_path)->tuple[dict, dict, list]:
	batch = []
	topics_dict = {}
	q_id_map = {}
	with open(topics_path, 'r', encoding="utf-8") as f:
		topics = json.load(f)
	for idx, dict in enumerate(topics):
		text = parseText(dict["Title"] + " " + dict["Body"])
		#expanded_text = text + " " + QueryExpansion.expand_query(text, len(text)*3)[0]
		topics_dict[dict["Id"]] = text
		q_id_map[idx] = dict["Id"]
		batch.append(text)
	return topics_dict, q_id_map, batch

def getDocs(docs_path)->tuple[dict, dict, list]:
	batch = []
	docs_dict = {}
	d_id_map = {}
	with open(docs_path, 'r', encoding="utf-8") as f:
		docs = json.load(f)
	for idx, dict in enumerate(docs):
		text = parseText(dict["Text"])
		docs_dict[dict["Id"]] = text
		d_id_map[idx] = dict["Id"]
		batch.append(text)
	return docs_dict, d_id_map, batch

def getBM25(queries, docs):
	bm25 = BM25Okapi(docs)
	results = []
	for query_batch in queries:
		for query in query_batch:
			results.append(bm25.get_scores(query))
	return results

# compute embeddings, save to file if provided
def getEmbeddings(batch, model):
	embeddings = model.encode(batch, batch_size=32, show_progress_bar=True)
	return embeddings
		
# given query/document embeddings, write top N most relevant documents for each query to an output file
def writeTopN(q_embs, d_embs, q_map:dict, d_map:dict, run_name:str, output_path:str, top_n:int = 100):
	similarities = cosine_similarity(q_embs, d_embs)
	with open(output_path, "w", newline = '') as f:
		writer = csv.writer(f, delimiter='\t')
		for i in range(len(q_embs)):
			top_indices = np.argsort(similarities[i])[::-1][:top_n]
			for rank, j in enumerate(top_indices):
				writer.writerow([q_map[i], 'Q0', d_map[j], rank+1, similarities[i][j], run_name])