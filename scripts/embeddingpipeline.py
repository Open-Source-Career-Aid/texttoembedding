from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# This function is used to compute the mean pooling of the token embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == '__main__':

    # load document chunks
    documentchunks = 'something'

    # load model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # iterate over chunks
    for chunk in documentchunks:

        # encode chunk
        encoded_input = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # normalize embeddings
        embedding = F.normalize(embeddings, p=2, dim=1)

        # convert to list
        embedding = [float(x) for x in embedding[0]]

        # save to index

        """

            'embedding' is a vector of 768 floats.

            Add code here to save the vectro to the index.

        """