#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:08:14 2021

@author: kanchan
"""

"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import init
import os


import sys
import math
import numpy as np
count_Encoder=0
sys.path.insert(0, './tools/densevid_eval/coco-caption') # Hack to allow the import of pycocoeval




INF = 1e10
def save_tensor(save_item,count_id,source_id, video_prefix=None):
    torch.save(save_item,'./save_tensor/'+source_id+str(count_id)+'.pt')  #./save_tensor4
    target_string = str(video_prefix)
    source_file = './save_tensor/'+source_id+str(count_id)+'.txt'
    file = open(source_file, "w+")
    file.write(target_string)
    return
  
def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1)).float()
        if x.is_cuda:
           positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())


    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)

def mask(targets, out):
    mask = (targets != 1)
    out_mask = mask.unsqueeze(-1).expand_as(out)
    return targets[mask], out[out_mask].view(-1, out.size(-1))

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(0.01*self.conv3(x))
        return x

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(1024, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 1)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x
class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
#        self_attn = matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)
#        print("4444444444",self_attn.shape)
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio,causal=False):
        super().__init__()
       
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads
        self.d_model=100
        self.drop_ratio=nn.Dropout(0.1)

        self.start=Network()
        self.end=Network()
       
        self.linear = nn.Linear(309,256)
        self.attention_weights_list = []
    def forward(self, query, key, value, video_prefix=None):
        global count_Encoder
        global count_Decoder_attn
        global count_Decoder_self
        
        query, key, value = self.wq(query), self.wk(key), self.wv(value)

        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        attention_outputs = []
        for q, k, v in zip(query, key, value):
            attention_weights = self.attention(q, k, v)
            attention_outputs.append(attention_weights)
            self.attention_weights_list.append(attention_weights)
        attention_out=self.wo(torch.cat([self.attention(q, k, v)
                          for q, k, v in zip(query, key, value)], -1))
       
        return attention_out,attention_outputs
       
    def save_attention_weights(self, file_path):
        # Save the attention weights to a file using a suitable serialization format
#        file_path = "/media/kanchan/Data/myfewshot/fewshotQAT/action_localization_visualization/attention_weights"
        for i, attention_weights in enumerate(self.attention_weights_list):
            weight_file_path = os.path.join(file_path, f'attention_weights_{i}.pt')
            torch.save(attention_weights, weight_file_path)
#        torch.save(self.attention_weights_list, file_path)
#      
class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x,y):
#      
        return self.feedforward(self.selfattn(x, y, y))

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio,source='Decoder_selfattn',causal=True),
            d_model, drop_ratio)
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio,source='Decoder_selfattn'),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)
        return self.feedforward(self.attention(x, encoding, encoding))

class Encoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
      
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x,y):
#      
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
#      
        encoding = []
        for layer in self.layers:
            x = layer(x,y)
#          
            encoding.append(x)
        return encoding[-1]

class Decoder(nn.Module):

    def __init__(self, d_model, d_hidden, vocab, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.out = nn.Linear(d_model, len(vocab))
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        self.vocab = vocab
        self.d_out = len(vocab)

    def forward(self, x, encoding):
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding):
            x = layer(x, enc)
        return x

    def greedy(self, encoding, T):
        B, _, H = encoding[0].size()
        # change T to 20, max # of words in a sentence
        # T = 40
        # T *= 2
        prediction = Variable(encoding[0].data.new(B, T).long().fill_(
            self.vocab.stoi['<pad>']))
        hiddens = [Variable(encoding[0].data.new(B, T, H).zero_())
                   for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(Variable(
                    encoding[0].data.new(B).long().fill_(
                        self.vocab.stoi['<init>'])), embedW)
            else:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(prediction[:, t - 1],
                                                                embedW)
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t + 1]
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l]))

            _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(-1)
        return hiddens, prediction


    def sampling(self, encoding, gt_token, T, sample_prob, is_argmax=True):
        B, _, H = encoding[0].size()
        # change T to 20, max # of words in a sentence
        # T = 40
        # T *= 2
        prediction = Variable(encoding[0].data.new(B, T).long().fill_(
            self.vocab.stoi['<pad>']))
        hiddens = [Variable(encoding[0].data.new(B, T, H).zero_())
                   for _ in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(Variable(
                    encoding[0].data.new(B).long().fill_(
                        self.vocab.stoi['<init>'])), embedW)
            else:
                use_model_pred = np.random.binomial(1, sample_prob, 1)[0]
                if use_model_pred > 0:
                    hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(
                        prediction[:, t - 1],
                        embedW)
                else:
                    hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(
                        gt_token[:, t], # t since gt_token start with init
                        embedW)
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t + 1]
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l]))

            if is_argmax:
                _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(-1)
            else:
                pred_prob = F.softmax(self.out(hiddens[-1][:, t]), dim=-1)
                prediction[:, t] = torch.multinomial(pred_prob,
                                                        num_samples=1,
                                                        replacement=True)
                prediction[:, t].detach_()

        return prediction


class Transformer(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=2, n_heads=2, drop_ratio=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
                               n_heads, drop_ratio)
        # self.decoder = Decoder(d_model, d_hidden, vocab_trg, n_layers,
        #                       n_heads, drop_ratio)

    def denum(self, data):
        return ' '.join(self.decoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '#').replace(' <pad>', '')

    def forward(self, x, video_prefix=None):

        encoding = self.encoder(x, video_prefix=video_prefix)
       

        return encoding[-1], encoding
       

class RealTransformer(nn.Module):

    def __init__(self, d_model, encoder, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1):
        super().__init__()
       
        self.encoder = encoder
        self.decoder = Decoder(d_model, d_hidden, vocab_trg, n_layers,
                              n_heads, drop_ratio)
        self.n_layers = n_layers


    def denum(self, data):
        return ' '.join(self.decoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '').replace(' <pad>', '').replace(' .', '').replace('  ', '')

    def forward(self, x, s, x_mask=None, sample_prob=0):
        encoding = self.encoder(x, x_mask)

        max_sent_len = 20
        if not self.training:
            if isinstance(s, list):
                hiddens, _ = self.decoder.greedy(encoding, max_sent_len)
                h = hiddens[-1]
                targets = None
            else:
                h = self.decoder(s[:, :-1].contiguous(), encoding)
                targets, h = mask(s[:, 1:].contiguous(), h)
            logits = self.decoder.out(h)
        else:
            if sample_prob == 0:
                h = self.decoder(s[:, :-1].contiguous(), encoding)
                targets, h = mask(s[:, 1:].contiguous(), h)
                logits = self.decoder.out(h)
            else:
                model_pred = self.decoder.sampling(encoding, s,
                                                   s.size(1) - 2,
                                                   sample_prob,
                                                   is_argmax=True)
                model_pred.detach_()
                new_y = torch.cat((
                    Variable(model_pred.data.new(s.size(0), 1).long().fill_(
                        self.decoder.vocab.stoi['<init>'])),
                    model_pred), 1)
                h = self.decoder(new_y, encoding)
                targets, h = mask(s[:, 1:].contiguous(), h)
                logits = self.decoder.out(h)

        return logits, targets

    def greedy(self, x, x_mask, T):
        encoding = self.encoder(x, x_mask)

        _, pred = self.decoder.greedy(encoding, T)
        sent_lst = []
        for i in range(pred.data.size(0)):
            sent_lst.append(self.denum(pred.data[i]))
        return sent_lst

    def scst(self, x, x_mask, s):
#        self.scorer = Meteor()
        encoding = self.encoder(x, x_mask)

        # greedy part
        _, pred = self.decoder.greedy(encoding, s.size(1)-1)
        pred_greedy = []
        for i in range(pred.data.size(0)):
            pred_greedy.append(self.denum(pred.data[i]))

        del pred
        # sampling part
        model_pred = self.decoder.sampling(encoding, s,
                                           s.size(1) - 2,
                                           sample_prob=1,
                                           is_argmax=False)
        model_pred.detach_()
        new_y = torch.cat((
            Variable(model_pred.data.new(s.size(0), 1).long().fill_(
                self.decoder.vocab.stoi['<init>'])),
            model_pred), 1)
        h = self.decoder(new_y, encoding)
        B, T, H = h.size()
        logits = self.decoder.out(h.view(-1, H)) #.view(B, T, -1)

        mask = (s[:,1:] != 1).float()
        _, pred_sample = torch.max(logits, -1)

        p_model = F.log_softmax(logits, dim=-1)
        logp = p_model[torch.arange(0,B*T).type(logits.data.type()).long(), pred_sample.data].view(B, T)

        pred_sample = pred_sample.view(B, T)

        assert pred_sample.size(0) == len(pred_greedy), (
            'pred_sample should have the same number of sentences as in '
            'pred_greedy, got {} and {} instead'.format(B, len(pred_greedy))
        )
        assert pred_sample.size() == (B, T), (
            'pred_sample size should error'
        )

        pred_sample.detach_()

        # rewards
        sentence_greedy, sentence_sample, sentence_gt = {}, {}, {}
        for i in range(len(pred_greedy)):
            sentence_greedy[i] = [{'caption':pred_greedy[i]}]
            sentence_sample[i] = [{'caption':self.denum(pred_sample.data[i])}]
            sentence_gt[i] = [{'caption':self.denum(s.data[i,1:])}]

        tok_greedy = self.tokenizer.tokenize(sentence_greedy)
        tok_sample = self.tokenizer.tokenize(sentence_sample)
        tok_gt = self.tokenizer.tokenize(sentence_gt)
        _, r_greedy = self.scorer.compute_score(tok_gt, tok_greedy)
        _, r_sample = self.scorer.compute_score(tok_gt, tok_sample)

        r_diff = [r_s-r_g for (r_s, r_g) in zip(r_greedy, r_sample)]
        r_diff = Variable(torch.Tensor(r_diff).type(logp.data.type()))

        loss = - torch.mean(torch.sum(r_diff.view(-1,1) * logp * mask, 1))

        return loss
