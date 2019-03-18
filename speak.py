import jieba
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("-l", "--load", help="optional argument", dest="filename", default="default")
args = parser.parse_args()
checkpoint_dir = "checkpoints/"

def save_checkpoint(state, is_best = False, filename='checkpoint.pth.tar'):
    torch.save(state, checkpoint_dir + filename)
    torch.save(state, checkpoint_dir + "checkpoint_lastest.pth.tar")
    if is_best:
        shutil.copyfile(filename, checkpoint_dir + 'model_best.pth.tar')

def test(start_idx, end_idx, sentence_len):
    with torch.no_grad():
        input_current = corpus[start_idx:end_idx]
        
        sentence = ""
        for w in input_current:
            sentence += w
        sample_length = 100
        
        input_current = prepare_sequence(input_current, w2i)   
        
        
        
        for _ in range(sample_length):
                
            tag_scores = model(input_current)
            values, indices = torch.max(tag_scores[-1], 0)           
            input_current = torch.Tensor(input_current.tolist()[1:] + [indices.tolist()])           
            sentence += recover_sequence([indices.tolist()], i2w)[0]
            input_current = prepare_sequence(recover_sequence(input_current.tolist(), i2w), w2i)

        
        random_start_idx = np.random.randint(len(corpus))
        input_current = corpus[random_start_idx]
        
        rand_sentence = ""
        for w in input_current:
            rand_sentence += w
        random_sample_length = 100
        
        input_current = prepare_sequence(input_current, w2i)
        
        max_input_length = sentence_len        
        
        for _ in range(random_sample_length):
                
            tag_scores = model(input_current)
            values, indices = torch.max(tag_scores[-1], 0)     
            
            #print("len(input_current)", len(input_current))
            
            if len(input_current) < max_input_length:
                input_current = torch.Tensor(input_current.tolist() + [indices.tolist()])
            else:
                input_current = torch.Tensor(input_current.tolist()[1:] + [indices.tolist()])           
            rand_sentence += recover_sequence([indices.tolist()], i2w)[0]
            input_current = prepare_sequence(recover_sequence(input_current.tolist(), i2w), w2i)

        print('------------------------------------------------')
        print("pred:", sentence)
        print('++++++++++++++++++++++++++++++++++++++++++++++++')
        print("real:", ''.join(corpus[start_idx:start_idx+sample_length+1]))
        print('================================================')
        print("rand:", rand_sentence)
        

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)       
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        
        embeds = self.word_embeddings(sentence)        
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))        
        vocab_space = self.linear(lstm_out.view(len(sentence), -1))
        vocab_scores = F.log_softmax(vocab_space, dim=1)
        return vocab_scores
        
def prepare_sequence(seq, w2i):
    idxs = [w2i[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
        
def recover_sequence(seq, i2w):
    words = [i2w[i] for i in seq]
    return words

w2num = defaultdict(int)
w2i = defaultdict(int)
i2w = {}
lines = []
corpus = []
with open("train.txt", "r") as f:
    for line in f:
        #line = line.strip()
        #if line == '':
        #    continue
        seg_list = jieba.cut(line)
        
        seg_list_list = []
        for word in seg_list:            
            w2num[word] += 1
            if word not in w2i:
                w2i[word] = len(w2i)
            seg_list_list += [word]
            corpus += [word]
            
        lines += [seg_list_list]

f.close()

for word, idx in w2i.items():
    i2w[idx] = word


print(w2i) 
print(i2w)

EMBEDDING_DIM = 64
HIDDEN_DIM = 64
print("EMBEDDING_DIM", EMBEDDING_DIM)
print("HIDDEN_DIM", HIDDEN_DIM)
print("vocab_size", len(w2i))

model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(w2i))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

if args.filename:
    if os.path.isfile(checkpoint_dir + args.filename):
        print("=> loading checkpoint '{}'".format(args.filename))
        checkpoint = torch.load(checkpoint_dir + args.filename)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        '''old_w2i = checkpoint['w2i']
        old_i2w = checkpoint['i2w']
        len_old_w2i = len(old_w2i)
        new_w2i = old_w2i
        new_i2w = {}
        for w_key, idx in w2i.items():
            if w_key not in old_w2i:
                new_w2i[w_key] = len(new_w2i)                
        
        for word, idx in new_w2i.items():
            new_i2w[idx] = word
            
        w2i = new_w2i
        i2w = new_i2w
        
        print("------------------------------")
        print(w2i) 
        print(i2w)
        print("===========================================")'''
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.filename, checkpoint['epoch']))
                      
    else:
        print("=> no checkpoint found at '{}'".format(args.filename))
        args.start_epoch = 0

with torch.no_grad():
    inputs = prepare_sequence(lines[0], w2i)
    tag_scores = model(inputs)

sentence_len = 10
start_idx = np.random.randint(len(corpus))
end_idx = start_idx + sentence_len
end_epoch = 10000000
if args.start_epoch + 1 >= end_epoch:
    print("args.start_epoch + 1 >= end_epoch")
    raise Exception

for epoch in range(args.start_epoch + 1, end_epoch):
    
    if end_idx+1 > len(corpus):
        sentence_len = 10
        start_idx = np.random.randint(len(corpus))
        end_idx = start_idx + sentence_len
        continue
         
    model.zero_grad()

    
    sentence_in = prepare_sequence(corpus[start_idx:end_idx], w2i)
    targets = prepare_sequence(corpus[start_idx+1:end_idx+1], w2i)
    vocab_scores = model(sentence_in)
    
    loss = loss_function(vocab_scores, targets)
    loss.backward()
    optimizer.step()
    
    start_idx = end_idx
    end_idx = start_idx + sentence_len
    
    if epoch % 1000 == 0:
        print('epoch:', epoch)
        test(start_idx, end_idx, sentence_len)
    if epoch % 100000 == 0:
        print('epoch:', epoch)
        test(start_idx, end_idx, sentence_len)
        save_checkpoint({
            'epoch': epoch,            
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'w2i' : w2i,
            'i2w' : i2w
        }, False, "checkpoint" + str(epoch) + ".pth.tar")

