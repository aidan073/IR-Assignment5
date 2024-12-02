import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.1-8B-Instruct"
token = "<>"
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token, load_in_8bit=True)
query_expansion_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def expand_query(queries, max_length):
    """
    Query expansion using Llama

    Args:
        query (str): Initial query
        max_length (int): Max length for the expanded query
        num_return_sequences (int): Number of expanded queries to generate

    Returns:
        A list of expanded queries
    """
    
    expansions = query_expansion_pipeline(
        queries,
        max_length=max_length,
        top_k=50,
        top_p=0.95,
        temperature=0.6,
        do_sample=True,
        truncation=True
    )
    
    return [expansion["generated_text"].strip() for expansion in expansions]

# example
# query = "Answer the following question: \"Why do airline tickets have titles in addition to names?\". Give the rationale before answering"
# expanded_queries = expand_query(query, 300)
# print("Original Query:", query)
# print("Expanded Queries:")
# for idx, expanded_query in enumerate(expanded_queries, 1):
#     print(f"{idx}. {expanded_query}")