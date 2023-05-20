
from transformers import AutoTokenizer, AutoModel
import torch

def get_batch_token_ids(batch, tokenizer):
    """Map batch to a tensor of ids. The return
    value should meet the following specification:

    1. The max length should be 512.
    2. Examples longer than the max length should be truncated
    3. Examples should be padded to the max length for the batch.
    4. The special [CLS] should be added to the start and the special 
       token [SEP] should be added to the end.
    5. The attention mask should be returned
    6. The return value of each component should be a tensor.    

    Parameters
    ----------
    batch: list of str
    tokenizer: Hugging Face tokenizer

    Returns
    -------
    dict with at least "input_ids" and "attention_mask" as keys,
    each with Tensor values

    """
    max_length = 512
    return tokenizer.batch_encode_plus(batch,
                                       add_special_tokens    = True,
                                       padding               = 'max_length',
                                       truncation            = True, 
                                       max_length            = max_length,
                                       return_tensors        = "pt",
                                       return_attention_mask = True
                                       )
    

def batch_iter(dataset_,batch_size):
    return (dataset_[pos:pos +batch_size] for pos in range(0, len(dataset_),batch_size))

def get_reps(dataset, model, tokenizer, batchsize=20,device="cpu"):
    """Represent each example in dataset with the final hidden state 
    above the [CLS] token.

    Parameters
    ----------
    dataset :   list of str
    model :     BertModel
    tokenizer : BertTokenizerFast
    batchsize : int

    Returns
    -------
    torch.Tensor with shape (n_examples, dim) where dim is the
    dimensionality of the representations for model

    """
    data = []
    model.to(device)
    with torch.no_grad():
        # Iterate over dataset in batches:
        for batch_list in batch_iter(dataset, batchsize):
            #print(batch_list )
            # Encode the batch with get_batch_token_ids:
            encoded_batch = get_batch_token_ids(batch_list, tokenizer)
            encoded_batch = {key: value.to(device) for key, value in encoded_batch.items()}

            # Get the representations from the model, making sure you pay attention to masking:
            model_output = model( encoded_batch['input_ids'], attention_mask=encoded_batch['attention_mask'])

            ### model_output is a dictionary with two keys 'last_hidden_state' which return the value of [batchsize,512,256] (512 is the tokenizer maximum input size, only mask values are of interest)
            ### 'pooler_output' is a linear head on  top of the last_hidden_state and it returns a tensor of shape  [batchsize,256]
            data.append( model_output['last_hidden_state'][:,0,:]) # only append the value of the cls token                                            


            #data.append(model_output ['pooler_output'])
            #print( model_output['last_hidden_state'].shape)
            #print(model_output ['pooler_output'].shape)
        # Return a single tensor:
        return  torch.vstack(data)#  model_output ['last_hidden_state']


class EntityEncoder:
    # The 'EntityEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='michiyasunaga/BioLinkBERT-base', device=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        if self.device is not None:
            self.model.to(self.device)
            self.model.eval()


    @torch.no_grad()
    def __call__(self, value):
        outputs = get_reps(value, self.model, self.tokenizer, device=self.device)
        return outputs

