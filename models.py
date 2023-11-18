import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# Use the GPU if it's available
use_gpu = torch.cuda.is_available()

# Quick utility function to sample from the Gumbel-distribution: -Log(-Log(Uniform)), eps to avoid numerical errors
def sample_gumbel(input_size, eps=1e-20):
    unif = torch.rand(input_size)
    if use_gpu:
        unif = unif.cuda()
    return -torch.log(-torch.log(unif + eps) + eps)

class Encoder(nn.Module):
    '''
    The encoder takes embedding dimension, M, and K (from MxK coding scheme introduced in paper)
    as parameters. From a word's baseline embedding,
    it outputs Gumbel-softmax or one-hot encoding vectors d_w, reshaped for the decoder.
    In the original paper, the hidden_layer was fixed at M * K/2.

    Input shape: BATCH_SIZE X EMBEDDING_DIM
    Output shape: BATCH_SIZE X M X K X 1, Code (K X 1)

    '''
    def __init__(self, emb_size,M, K, hidden_size= None):
        super(Encoder, self).__init__()
        # For GloVE vectors, emb_size = 300
        self.emb_size = emb_size
        self.K = K
        self.M = M
        # If not otherwise specified, use hidden_size indicated by paper
        if not hidden_size:
            hidden_size = int(M * K / 2)
        # This linear layer maps to latent hidden representation
        self.h_w = nn.Linear(self.emb_size, hidden_size)
        # This layer maps from the hidden layer to BATCH_SIZE X M K
        self.alpha_w = nn.Linear(hidden_size, M * K)

    def forward(self, x, tau=1, eps=1e-20, training=True):
        # We apply hidden layer projection from original embedding
        hw = F.tanh(self.h_w(x))
        # We apply second projection and softplus activation
        alpha = F.softplus(self.alpha_w(hw))
        # This rearranges alpha to be more intuitively BATCH_SIZE X M X K
        alpha = alpha.view(-1, self.M, self.K)
        # Take the log of all elements
        log_alpha = torch.log(alpha)
        # We apply Gumbel-softmax trick to get code vectors d_w
        d_w = F.softmax((log_alpha + sample_gumbel(log_alpha.size())) / tau, dim=-1)
        # Find argmax of all d_w vectors
        _, ind = d_w.max(dim=-1)
        if not training:
            # Allows us when not training to convert soft vector to a hard, binarized one-hot encoding vector
            d_w = torch.zeros_like(d_w).scatter_(-1, ind.unsqueeze(2), 1.0)
        # d_w is now BATCH x M x K x 1
        d_w = d_w.unsqueeze(-1)
        return d_w, ind

class Decoder(nn.Module):
    '''
    The decoder receives d_w as input from the encoder, and outputs the embedding generated by this code.
    It stores a set of source dictionaries, represented by A, and computes the proper embedding from a summation
    of M matrix-vector products.

    INPUT SHAPE: BATCH_SIZE X M X K X 1
    OUTPUT SHAPE: BATCH_SIZE X EMBEDDING_DIM
    '''
    def __init__(self, M, K, output_size):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.K = K
        self.M = M
        # Contains source dictionaries for computing embedding given codes
        self.A = Source_Dictionary(M, output_size, K)

    # Following the formula in the paper, performs multiplication and summation over the M matrix-vector products
    def forward(self, d_w):
        output = self.A(d_w)
        output = torch.sum(output, dim=1)
        return output


class Code_Learner(nn.Module):
    '''
    Our final autoencoder model structure used to train our encoded embeddings with an end-to-end network
    INPUT: baseline embeddings BATCH_SIZE X EMB_DIMENSIONS
    OUTPUT: BATCH_SIZE X EMBEDDING_DIM (final encoding representation)


    '''
    def __init__(self,emb_size, M, K, hidden_size = None):
        super(Code_Learner, self).__init__()
        # Initialize encoder and decoder components
        self.encoder = Encoder(emb_size,M, K, hidden_size)
        self.decoder = Decoder(M, K, emb_size)

    # Set up basic, normal encoder-decoder structure
    def forward(self, x, tau=1, eps=1e-20, training=True):
        d_w, _ = self.encoder(x, tau, eps, training)
        comp_emb = self.decoder(d_w)
        return comp_emb


class Source_Dictionary(nn.Module):
    r"""I basically modified the source code for the nn.Linear() class
        Removed bias, and the weights are of dimension M X EMBEDDING_SIZE X K
        INPUT: BATCH_SIZE X M X K X 1

        OUTPUT:BATCH_SIZE X K X 1
    """

    def __init__(self, M, emb_size, K):
        super(Source_Dictionary, self).__init__()
        # The weight of the dictionary is the set of M dictionaries of size EMB X K
        self.weight = Parameter(torch.Tensor(M, emb_size, K))
        self.reset_parameters()

    # Initialize parameters of Source_Dictionary
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    # Operation necessary for proper batch matrix multiplication
    def forward(self, input):
        result = torch.matmul(self.weight, input)
        return result.squeeze(-1)


class Classifier(nn.Module):
    '''
    Implementation of a Sentiment Classifier as specified by the paper

    '''
    def __init__(self, embedding, batch_size, hidden_size=150):
        super(Classifier, self).__init__()
        # Infer vocab, embedding_dimension size by the given embedding
        self.vocab_size, self.emb_size = embedding.size()
        # Initialize and copy weights from given embedding
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.embedding.weight.data.copy_(embedding)
        # Note that these weights will be fixed
        self.embedding.weight.requires_grad=False
        # Single LSTM layer
        self.lstm = nn.LSTM(input_size = self.emb_size, hidden_size = hidden_size)
        # Linear layer projected to 1x2 vector (positive / negative)
        self.fc = nn.Linear(hidden_size, 2)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        # Used for initializing cell and hidden states for LSTM layer
        self.c0 = torch.zeros(1, 1, hidden_size, requires_grad=True)
        self.h0 = torch.zeros(1, 1, hidden_size, requires_grad=True)

    def forward(self, x):
        # Construct cell and hidden states for LSTM layer, in a way where batch size does not have to be specified
        h = torch.cat([self.h0 for _ in range(x.size(1))], 1)
        c = torch.cat([self.c0 for _ in range(x.size(1))], 1)
        # If using GPU, put into CUDA memory
        if use_gpu:
            h, c = h.cuda(), c.cuda()
        # Embed review into encoding
        x = self.embedding(x)
        # Pass to LSTM layer
        out, _ = self.lstm(x, (h, c))
        # Project with fc layer
        out = self.fc(out[-1])
        # Take log_softmax
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
