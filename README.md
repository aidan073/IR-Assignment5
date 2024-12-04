# This repo was used for testing the performance of LLAMA3.1-Instruct-8B on query expansion.  
The expanded queries were tested downstream on retrieval with gte-large-en-v1.5. 
  
## Installation:  
  
- clone the repo  
- run `pip install -r requirements.txt`  
- obtain an access token for LLAMA3.1  
- replace `token = None` in QueryExpansion.py with the obtained access token  
- place a Queries.json and Documents.json file into the cloned repo directory  
- both files must be an array of JSON objects (each being a query/document). Queries.json must have fields "Title", "Body", and "Id". Documents.json must have fields "Id" and "Text".
- see usage to run  

## Usage:
  
`python initialResults.py <queries.json> <documents.json> <desired_results_filename.tsv>`