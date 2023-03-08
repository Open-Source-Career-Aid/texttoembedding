from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import re
import json

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':

    chunk_dir = 'chunks/'  

    result = list()

    hid = 0

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

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


                encoded_input = tokenizer(chunk_txt, padding=True, truncation=True, return_tensors='pt')

                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)

                 # Perform pooling
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                embedding = F.normalize(embeddings, p=2, dim=1)
                embedding = [float(x) for x in embedding[0]]
                print(list(embedding), '\n\n\n')
                print(doc_id, hid, '\n\n\n')
                info = {'document': doc_id, 'heading':head_id, 'vector': list(embedding)}
                result.append(info)
    with open('index_mpnet.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)