import torchtext
import os
import torch.optim as optim
import torch
import numpy as np
import math
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchtext.legacy import data
from torchtext.legacy import datasets

from models import Code_Learner, Classifier
from utils import test_model

import time

# Use the GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='Train a classifier to compare performance between GloVE and encoding')
# Directory where we want to write everything we save in this script to
parser.add_argument('--data_folder', default='data/', metavar='DIR',
                    help='folder to retrieve embeddings, data, text files, etc.')
parser.add_argument('--models_folder', default='models/', metavar='DIR',help='folder to save models')
parser.add_argument('--model_file', default='epoch_147000.pt', metavar='DIR',help='specific directory to model you want to load')
parser.add_argument('--embedding_size', default=300, type=int, metavar='N', help='Embedding dimension size, default: 300')
parser.add_argument('--M', default=32, type=int, metavar='N', help='Number of source dictionaries, default: 64')
parser.add_argument('--K', default=16, type=int, metavar='N', help='Source dictionary size, default: 8')
parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='Adam learning rate, default: 0.0001')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='Minibatch size, default: 128')
parser.add_argument('--max_len', default=400, type=int, metavar='N', help='Max sentence length allowed, default: 400')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='Total number of epochs, default: 30')
parser.add_argument('--embedding_type', default='coded', metavar='MODEL_TYPE', help='Model type; pick either coded or baseline, default: coded')


def classifier_train(epochs, classifier, optimizer, loss_func, train_iter, test_iter, c_type):
    classifier.train()
    # Initialize validation loss
    best_val_loss = float('inf')
    sdir = os.path.join(args.models_folder, '%s_classifier'%c_type)
    if not os.path.isdir(sdir):
        print("create directory for saving models: %s"%sdir)
        os.mkdir(sdir)
    # For every epoch
    for epoch in range(epochs):
        valid_loss = 0
        # For every batch
        classifier.train()
        for batch in train_iter:
            # Each batch has review, label
            data, label = batch.text, batch.label
            if use_gpu:
                data, label = data.to(device), label.to(device)
            # Labels in original dataset are given as 1 and 2, so we make that 0 and 1 instead
            label = label - 1
            # Clear gradients
            optimizer.zero_grad()
            # Make predictions with our _classifier
            preds = classifier(data)
            # Calculate loss
            loss = loss_func(preds, label)
            # Compute sum of gradients
            loss.backward()
            # Perform optimization step
            optimizer.step()
        train_loss, train_acc = test_model(classifier, train_iter,loss_func,device)
        valid_loss, valid_acc = test_model(classifier, train_iter,loss_func,device)
        
        # If this is our lowest validation loss, save the model
        if valid_loss < best_val_loss:
            spath = os.path.join(sdir, 'epoch_%s.pt'%(epoch))
            print("save mode in %s"%spath)
            torch.save(classifier, spath)
            best_val_loss = valid_loss
        # Calculate accuracy and report
        print('''Epoch [{e}/{num_e}]\t Loss: {l_t:.3f}/{l_v:.3f} Accuracy: {r_t:.3f}/{r_v:.3f}'''.format(e=epoch+1, num_e=epochs, l_t=train_loss, 
                                                                                                         l_v = valid_loss, r_t =train_acc, r_v = valid_acc))

def main():
    global args
    # Parse commands from ArgumentParser
    args = parser.parse_args()
    args.models_folder = os.path.join(args.models_folder,'%s_%s'%(args.M,args.K))
    args.model_file = os.path.join(args.models_folder,args.model_file)
    # Our text field for imdb data
    TEXT = data.Field(lower=True)
    # Our label field for imdb data
    LABEL = data.Field(sequential=False)
    # Load GloVE embeddings
    orig_embeddings = torch.load(args.data_folder + 'all_orig_emb.pt')
    total_words = len(orig_embeddings)
    # Load shared words and all GloVE words
    with open(args.data_folder + "shared_words.txt", "r") as file:
        shared_words = file.read().split('\n')
    with open(args.data_folder + "glove_words.txt", "r") as file:
        glove_words = file.read().split('\n')
    # Recreate GloVE_dict
    glove_dict = {}
    for i, word in enumerate(glove_words):
        glove_dict[word] = orig_embeddings[i]

    # Load IMDB dataset with standard splits and restrictions identical to paper
    train, test = datasets.IMDB.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral' and len(ex.text) <= 400)

    # Both loops go through the words of train and test dataset, finds words without glove vectors, and replaces them with <unk>
    for i in range(len(train)):
        review = train.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'
    for i in range(len(test)):
        review = test.examples[i].text
        for i, word in enumerate(review):
            if word not in glove_dict:
                review[i] = '<unk>'

    # Build modified vocabulary
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # Create iterators over train and test set
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=args.batch_size, repeat=False, device=-1)

    # If we want to use baseline GloVE embeddings
    if args.embedding_type == 'baseline':
        # Initialize embedding
        comp_embedding = np.random.uniform(-0.25, 0.25, (len(TEXT.vocab), args.embedding_size))
        # For each vocab word, replace embedding vector with GloVE vector
        for word in shared_words:
            comp_embedding[TEXT.vocab.stoi[word]] = glove_dict[word]
        # Initialize Classifer with our GloVE embedding
        base_c = Classifier(torch.FloatTensor(comp_embedding), args.batch_size)
        # Put model into CUDA memory if using GPU
        if use_gpu:
            base_c = base_c.to(device)
        # Initialize Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,base_c.parameters()), lr=args.lr)
        # Define Loss function
        loss_func = nn.NLLLoss()

    else:
        '''
        Note- the model in the paper is different because they only store the source dictionaries,
        making their model smaller than normal classifiers which is a major purpose of the paper.
        By my formulation, my model actually has the same size. However, they are fundamentally equivalent,
        except that the authors would have to preprocess the data (convert words into codes) whereas I
        simply make an embedding layer of size Vocab like GloVE vectors. Either way, I should get the same
        levels of accuracy, which is the primary importance of the sentiment classification task- to check
        whether the coding embeddings still give the same level of accuracy.
        '''
        # Initialize embedding
        code_embedding = torch.FloatTensor(np.random.uniform(-0.25, 0.25, (len(TEXT.vocab), args.embedding_size)))
        # Load best model for code embedding generation
        model = Code_Learner(args.embedding_size, args.M, args.K)
        model = torch.load(args.model_file)
        # Put model into CUDA memory if using GPU
        if use_gpu:
            code_embedding = code_embedding.to(device)
            model = model.to(device)
        # For all words in vocab
        for i in range(len(TEXT.vocab)):
            # Try to see if it has a corresponding glove_vector
            try:
                glove_vec = glove_dict[TEXT.vocab.itos[i]]
                if use_gpu:
                    glove_vec = glove_vec.to(device)
                # If so, then generate our own embedding for the word using our model
                code_embedding[i] = model(glove_vec, training=False)
            # The word doesn't have a GloVE vector, keep it randomly initialized
            except KeyError:
                pass
        base_c = Classifier(torch.FloatTensor(code_embedding.cpu()), args.batch_size)
        # Put model into CUDA memory if using GPU
        if use_gpu:
            base_c = base_c.to(device)
        # Initialize Optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,base_c.parameters()), lr=args.lr)
        # Define Loss function
        loss_func = nn.NLLLoss()
    st = time.time()
    classifier_train(args.epochs, base_c, optimizer, loss_func, train_iter, test_iter, args.embedding_type)
    et = time.time()
    print("Elapsed time for training: %s"%(et-st))
if __name__ == '__main__':
    main()
