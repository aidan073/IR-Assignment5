import csv
import json
import re
from rank_bm25 import BM25Okapi
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

def parseText(text, word_limit=None)->str:
    text = BeautifulSoup(text, "html.parser").text
    return ' '.join(re.sub(r"\\'", "'", text).split()[0:word_limit]) # replace noise character and limit query lengths

# take the original qrel, and create a subset qrel containing only those that appear in queries (used for testing slow models)
def filter_qrel(queries:dict, input_qrel_path: str, output_qrel_path: str):
    filter_ids = []
    for idx, sample_id in queries.items():
        filter_ids.append(sample_id)
    filter_ids_set = set(filter_ids)

    with open(input_qrel_path, "r") as infile, open(output_qrel_path, "w") as outfile:
        for line in infile:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            
            qid = parts[0]
            if qid in filter_ids_set:
                outfile.write(line)

def getTopics(topics_path)->tuple[dict, dict, list]:
    batch = []
    topics_dict = {}
    q_id_map = {}
    with open(topics_path, 'r', encoding="utf-8") as f:
        topics = json.load(f)
    for idx, dict in enumerate(topics):
        full_text = parseText(dict["Title"] + " " + dict["Body"])
        topics_dict[dict["Id"]] = full_text
        q_id_map[idx] = dict["Id"]
        batch.append(full_text)
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

# compute bm25 results for a list of queries and docs
def getBM25(queries, docs):
    bm25 = BM25Okapi(docs)
    results = []
    for query_batch in queries:
        for query in query_batch:
            results.append(bm25.get_scores(query))
    return results

# compute embeddings
def getEmbeddings(batch, model):
    embeddings = model.encode(batch, batch_size=10, show_progress_bar=True)
    return embeddings

# save expaned queries
def saveNewQueries(queries, output_path):
    with open(output_path, "w") as f:
        for idx, query in enumerate(queries):
            f.write(f"{idx}: {query}\n")

# save expaned docs
def saveNewDocs(modified_docs, doc_id_map, original_docs, output_path):
    with open(output_path, "w") as f:
        for i in range(len(modified_docs)):
            original_docs[doc_id_map[i]]["Text"] = modified_docs[i]
        json.dump(original_docs, f, ensure_ascii=False)
        
# given query/document embeddings, write top N most relevant documents for each query to an output file
def writeTopN(q_embs, d_embs, q_map:dict, d_map:dict, run_name:str, output_path:str, top_n:int = 100, normalized=False):
    if not normalized:
        similarities = cosine_similarity(q_embs, d_embs)
    else:
        similarities = q_embs @ d_embs.T
    with open(output_path, "w", newline = '') as f:
        writer = csv.writer(f, delimiter='\t')
        for i in range(len(q_embs)):
            top_indices = np.argsort(similarities[i])[::-1][:top_n]
            for rank, j in enumerate(top_indices):
                writer.writerow([q_map[i], 'Q0', d_map[j], rank+1, similarities[i][j], run_name])