import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.1-8B-Instruct"
token = ""
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, token=token)
query_expansion_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def expand_query(queries, max_length, num_return_sequences=1):
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
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        truncation=True
    )
    
    return [expansion["generated_text"].strip() for expansion in expansions]

# example
query = "Why do airline tickets have titles in addition to names?"
expanded_queries = expand_query(query, 100)
print("Original Query:", query)
print("Expanded Queries:")
for idx, expanded_query in enumerate(expanded_queries, 1):
    print(f"{idx}. {expanded_query}")