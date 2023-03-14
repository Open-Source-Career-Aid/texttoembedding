from transformers import AutoModel, AutoTokenizer
import torch
path = path = 'roberta-large'

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path,output_hidden_states=True)

if __name__ == '__main__':
    text_file = "/content/eg.txt"
    file = open(text_file,"r")  
    file_name = text_file.split(".")[0]
    txt  = "".join(file.readlines())

    tokens = tokenizer.encode_plus(txt, add_special_tokens=False)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    # define our starting position (0) and window size (number of tokens in each chunk)
    start = 0
    window_size = 512
    total_len = len(input_ids)
    # initialize condition for our while loop to run
    loop = True
    # loop through and print out start/end positions
    while loop:
        # the end position is simply the start + window_size
        end = start + window_size
        # if the end position is greater than the total length, make this our final iteration
        if end >= total_len:
            loop = False
            # and change our endpoint to the final token position
            end = total_len
        #print(f"{start=}\n{end=}")
        # we need to move the window to the next 512 tokens
        start = end
    # initialize probabilities list
    embed_list = []

    start = 0
    window_size = 510  # we take 2 off here so that we can fit in our [CLS] and [SEP] tokens

    loop = True

    while loop:
        end = start + window_size
        if end >= total_len:
            loop = False
            end = total_len
        # (1) extract window from input_ids and attention_mask
        input_ids_chunk = input_ids[start:end]
        attention_mask_chunk = attention_mask[start:end]
        # (2) add [CLS] and [SEP]
        input_ids_chunk = [101] + input_ids_chunk + [102]
        attention_mask_chunk = [1] + attention_mask_chunk + [1]
        # (3) add padding upto window_size + 2 (512) tokens
        input_ids_chunk += [0] * (window_size - len(input_ids_chunk) + 2)
        attention_mask_chunk += [0] * (window_size - len(attention_mask_chunk) + 2)
        # (4) format into PyTorch tensors dictionary
        input_dict = {
            'input_ids': torch.Tensor([input_ids_chunk]).long(),
            'attention_mask': torch.Tensor([attention_mask_chunk]).int()
        }

        with torch.no_grad():
            states = model(**input_dict).hidden_states
        #output = torch.stack([states[i] for i in range(len(states)-1, len(states))])
        output = states[-1]
        output = output.squeeze()
        #output = torch.mean(output, dim=0)
        #print(output)
        embed_list.append(output)

        start = end
    output = torch.mean(torch.stack(embed_list),0)
    outfile=str(file_name)+".pt"
    torch.save(output,outfile)
    # output size (512, 1024)
