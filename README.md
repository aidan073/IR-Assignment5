# This repo was used for testing the performance of LLAMA3.1-Instruct-8B on query expansion.  
Additional DocExpansion code provided, but was not implemented in my testing due to hardware constraints. **See IR-Assignment5 Analysis.pdf for experiment results.**
  
## Installation:  
  
- clone the repo  
- run `pip install -r requirements.txt`  
- place a Queries.json and Documents.json file into the cloned repo directory  
- both files must be an array of JSON objects (each object being a query or document). Queries.json must have fields "Title", "Body", and "Id". Documents.json must have fields "Id" and "Text".
- see usage to run  

## Usage:
  
`python initialResults.py <queries.json> <documents.json> <desired_results_filename.tsv>`
