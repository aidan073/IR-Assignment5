import data
import torch
import argparse
import QueryExpansion
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Initial ranks")

parser.add_argument("topics_path", type=str, help="Path to the topics file")
parser.add_argument("docs_path", type=str, help="Path to the documents file")
parser.add_argument("outfile_name", type=str, help="Desired output file name")

args = parser.parse_args()

docs_path = args.docs_path
topics_path = args.topics_path
outfile_name = args.outfile_name

topic_dict, q_id_map, q_batch = data.getTopics(topics_path)
doc_dict, d_id_map, d_batch = data.getDocs(docs_path)
q_batch = [f"Answer the following question: \"{query}\". Give the rationale before answering." for query in q_batch]
batch_size = 10
q_batches = [q_batch[i:i + batch_size] for i in range(0, len(q_batch), batch_size)]
expanded_queries = []
for idx, batch in enumerate(q_batches, 1):
    print(f"Starting batch {idx}")
    finalized_queries = QueryExpansion.expand_query(batch, 512)
    # finalized_queries = [topic_dict[q_id_map[(i)]].strip() + " " + query for i, query in enumerate(finalized_queries)]
    for query in finalized_queries:
        expanded_queries.append(finalized_queries)
    print(f"Expanded batch {idx} out of {len(q_batches)}")

# run initial model top get top 100s
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=device)
q_embs = data.getEmbeddings(expanded_queries, model)
d_embs = data.getEmbeddings(d_batch, model)
# scores = data.getBM25(expanded_queries, d_batch)
data.writeTopN(q_embs, d_embs, q_id_map, d_id_map, "bi_encoder", outfile_name)
