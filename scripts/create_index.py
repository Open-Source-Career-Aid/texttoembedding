import openai
import os
import re
import json

# enter openai api key here
openai.api_key = None

# takes in text and returns a vector embedding, i.e. a list of n-dimensions
def gpt3_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


if __name__ == '__main__':

	chunk_dir = 'chunks/'  # path to directory containing chunk files

	result = list()

	hid = 0
	for filename in os.listdir(chunk_dir):
	    hid+=1
	    filepath = os.path.join(chunk_dir, filename)  # full path to current file
	    if os.path.isfile(filepath):  # make sure it's a file, not a directory
	        # extract IDs from filename using regular expressions
	        match = re.search(r'text_chunk_(\d+)_document_(\d+)_head_(\d+)', filename)
	        if match:
	            chunk_id = int(match.group(1))
	            doc_id = int(match.group(2))
	            head_id = int(match.group(3))
	            
	            if doc_id==501:
	            	break

	            with open(filepath, 'r') as f:
	                chunk_txt = f.read()
	            
	            embedding = gpt3_embedding(chunk_txt.encode(encoding='ASCII',errors='ignore').decode())
	            print(embedding, '\n\n\n')
	            print(doc_id, hid, '\n\n\n')
	            info = {'document': doc_id, 'heading':head_id, 'vector': embedding}
	            result.append(info)
	with open('index.json', 'w') as outfile:
	    json.dump(result, outfile, indent=2)