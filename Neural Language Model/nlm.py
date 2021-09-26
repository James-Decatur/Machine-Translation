# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

# Professor: Gongbo Tang
# Assignment 3 - Neural Language Models


def data2batch(data, batch_size):
    ''' transform data into batches '''
    # figure out how cleanly we can divide the data into batches
    num_batch = data.size(0) // batch_size
    # trim off any extra elements that not fit
    data = data.narrow(0, 0, num_batch * batch_size)
    # divide the data into batches
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(hidden):
    ''' wraps hidden states into a new tensor, to detach them from the history '''
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(vec) for vec in hidden)


def get_batch(source, i):
    seq_len = min(args.length_seq, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+seq_len+1].view(-1)
    return data, target


def evaluate(data_source):
    # evaluation mode, no dropout
    model.eval()
    total_loss = 0.0
    num_tokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.length_seq):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # training mode, enables dropout
    model.train()
    total_loss = 0.0
    start_time = time.time()
    num_tokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    #optimizer.zero_grad()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.length_seq)):
        ''' we train models in the mini-batch style, i.e., update the model every batch '''
        # batch: the index of all the batches
        # i: the index of all the sentences
        data, targets = get_batch(train_data, i)
        model.zero_grad()  # need to reset the gradients in each batch
        hidden = repackage_hidden(hidden)
        # TODO: given the data, compute the gradients to update the model
        #  1) forward computation 2) compute the loss 3) backpropagation -> gradients
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        # clip_grad_norm helps to prevent the gradient explosion in RNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        #print(model.parameters())
        
        for p in model.parameters():
            # TODO: update model parameters based on gradients
            p.data.add_(p.grad.data, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} |ms/batch {:5.2f} |'
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.length_seq, lr,
                elapsed * 1000 /args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN-based Language Model')
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emb_size', type=int, default=25,
                        help='size of word embeddings')
    parser.add_argument('--num_hidden', type=int, default=25,
                        help='number of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--length_seq', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')

    parser.add_argument('--dry_run', action='store_true',
                        help='verify the code and the model')

    args = parser.parse_args()

    # set the random seed manually for reproducibility
    # for both cpu and gpu
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: you have GPU devices, you should run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    ############################################################
    # load data
    ############################################################

    corpus = data.Corpus(args.data)

    eval_batch_size = 20
    train_data = data2batch(corpus.train, args.batch_size)
    dev_data = data2batch(corpus.dev, args.batch_size)
    test_data = data2batch(corpus.test, args.batch_size)

    ############################################################
    # create the model
    ############################################################

    # load the dictionary
    num_tokens = len(corpus.dictionary)
    model = model.RNNModel(args.model, num_tokens, args.emb_size,
                           args.num_hidden, args.num_layers, args.dropout,
                           args.tied).to(device)

    # Define the loss function
    criterion = nn.NLLLoss()

    ############################################################
    # training
    ############################################################

    # loop over epochs
    lr = args.lr
    best_val_loss = None

    # you can use ctrl+c to break the training
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            if args.dry_run:
                break
            val_loss = evaluate(dev_data)
            print("-" * 88)
            # save the model if the validation loss is the best so far
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # the model is not further improved, reduce the learning rate
                # (smaller step, learn slower)
                lr /= 2.0
    except KeyboardInterrupt:
        print("-" * 88)
        print("Exiting from training early")

    '''
    # Load the best saved model
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        model.rnn.flatten_parameters()

    # testing
    test_loss = evaluate(test_data)
    print("-" * 88)
    print('|end of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print("-" * 88)
    '''



