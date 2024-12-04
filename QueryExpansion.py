import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.1-8B-Instruct"
token = None
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token, load_in_8bit=True)
query_expansion_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def expand_query(queries, max_new_tokens):
    """
    Query expansion using Llama

    Args:
        query (str): Initial query
        max_new_tokens (int): Max number of tokens to be generated

    Returns:
        A list of expanded queries
    """
    
    expansions = query_expansion_pipeline(
        queries,
        max_new_tokens=max_new_tokens,
        top_k=20,
        top_p=0.90,
        temperature=0.6,
        do_sample=True,
        truncation=True
    )
    
    return [expansion[0]["generated_text"].strip() for expansion in expansions]