import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base import BaseModel
import esm

class ESM2Embedding(nn.Module):
    # Code was adpated from NetSurfP-3.0: https://github.com/Eryk96/NetSurfP-3.0/blob/main/nsp3/nsp3/embeddings/esm1b.py
    """ ESM2 embedding layer module """

    def __init__(self, embedding_args: dict, embedding_pretrained=None, ft_embed_tokens: bool = False, ft_transformer: bool = False, ft_contact_head: bool = False,
                 ft_embed_positions: bool = False, ft_emb_layer_norm_before: bool = False, ft_emb_layer_norm_after: bool = False, 
                 ft_lm_head: bool = False):
        """ Constructor
        Args:
            embedding_args: arguments to embeddings model
            embedding_pretrained: patht to pretrained model
            ft_embed_tokens: finetune embed tokens layer
            ft_transformer: finetune transformer layer
            ft_contact_head: finetune contact head
            ft_embed_positions: finetune embedding positions
            ft_emb_layer_norm_before: finetune embedding layer norm before
            ft_emb_layer_norm_after: finetune embedding layer norm after
            ft_lm_head: finetune lm head layer
        """
        super(ESM2Embedding, self).__init__()

        # if given model path then pretrain
        if embedding_pretrained:
            self.model, _ = esm.pretrained.load_model_and_alphabet_local(embedding_pretrained)
        else:
            # configure pre-trained model
            alphabet = esm.Alphabet.from_architecture(embedding_args['arch'])
            model_type = esm.ProteinBertModel
            self.model = model_type(Namespace(**embedding_args), alphabet,)

        # finetuning, freezes all layers by default since every 
        self.finetune = [ft_embed_tokens, ft_transformer, ft_contact_head,
            ft_embed_positions, ft_emb_layer_norm_before, ft_emb_layer_norm_after, ft_lm_head]
        # print("finetune status of all layers", self.finetune)

        # finetune by freezing unchoosen layers
        for i, child in enumerate(self.model.children()):
            if self.finetune[i] == False:
                # print("layer "+str(i)+", <", child, "> is frozen!!!")
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, batch_tokens: torch.tensor, padding_length: int = None) -> torch.tensor:
        """ Convert tokens to embeddings
        Args:
            batch_tokens: tensor with sequence tokens
        """
        batch_residues_original = batch_tokens.shape[1] # i.e. max_seq_len+2

        # remove padding, make sure each token is less than padding_length
        if padding_length:
            batch_tokens = batch_tokens[:, :padding_length]

        batch_residues = batch_tokens.shape[1]
        
        # shape=batch_size, max_emdebbding, 1280
        embedding = self.model(batch_tokens[:, :], repr_layers=[33])["representations"][33]
        
        # convert nan to num 0.0
        embedding[embedding != embedding] = 0.0

        # cleanup
        del batch_tokens
        torch.cuda.empty_cache()

        # return embedding[:, 0:embedding.shape[1], :] # batch_size, 1024, 1280
        return embedding[:, 1:embedding.shape[1]-1, :] # batch_size, 1022, 1280, since <cls> and <seg> are useless

class Baseline(BaseModel):
    def __init__(self, n_hidden, embedding_args: dict, embedding_pretrained: str = None, **kwargs):
        """ Constructor
        Args:
            init_n_channels: size of the incoming feature vector
            n_hidden: amount of hidden neurons
            n_hidden1: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super().__init__()

        # ESM2 block
        self.embedding = ESM2Embedding(embedding_args, embedding_pretrained, **kwargs)

        # loss function will do log_softmax inside, no need to add a softmax layer here.
        # but remember to convert the output to probability by using: torch.exp(F.log_softmax(input, dim=1))
        self.net = nn.Sequential(*[
        nn.Linear(1280, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 128),
        nn.ReLU(),
        nn.Linear(128,2)])
        
    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic
        Args:
            x: batch_tokens (tensor)
            mask: valid_lengths of batch_tokens (tensor)
        """
        x = self.embedding(x, max(mask)+2)
        # average residual embeddings to seq embeddings
        x = x.mean(axis=1)
        # output shape=(batch_size, seq_len, 2)
        x = self.net(x)
        return x
