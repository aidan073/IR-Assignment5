import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("BeIR/query-gen-msmarco-t5-large-v1")
model = T5ForConditionalGeneration.from_pretrained("BeIR/query-gen-msmarco-t5-large-v1")
model.eval()

def docExpansion(docs:list[str], num_return_sequences:int=3):
    input_ids_batch = tokenizer(docs, return_tensors="pt", padding=True, truncation=True).input_ids

    with torch.no_grad():
        outputs_batch = model.generate(
            input_ids=input_ids_batch,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=num_return_sequences
        )
    
        decoded_outputs = tokenizer.batch_decode(outputs_batch, skip_special_tokens=True)
        
        expanded_docs = []
        for idx, doc in enumerate(docs):
            temp = decoded_outputs[idx * num_return_sequences:(idx + 1) * num_return_sequences]
            expanded_docs.append(' '.join(temp) + doc)

    return expanded_docs