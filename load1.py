#encoding:utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
torch.manual_seed(1)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    _,idx=torch.max(vec,1)
    return to_scalar(idx)
    
def prepare_sequence(seq,to_ix):
    #print ' '.join(seq)
    idxs=[to_ix[w] for w in seq]
    tensor=torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def log_sum_exp(vec):
    max_score=vec[0,argmax(vec)]
    max_score_broadcast=max_score.view(1,-1).expand(1,vec.size()[1])
    return max_score+\
            torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))
#bilstm_crf=torch.load('model3.pkl')#,'/home/pc/桌面/ner/model.pkl')
START_TAG="<START>"
STOP_TAG="<STOP>"
EMBEDDING_DIM=100
HIDDEN_DIM=100

tag_to_ix={"B":0,"I":1,"O":2,START_TAG:3,STOP_TAG:4}
ix_to_tag={0:"B",1:"I",2:"O",3:START_TAG,4:STOP_TAG}

#分数据:
def tolines(srcf):
    return open(srcf,"r").read().strip().decode('utf-8').split('\n')
import random
from sklearn import cross_validation
lines=tolines(u'data//label-人民日报.txt')[:100]
train_lines,dev_lines = cross_validation.train_test_split(lines,test_size=0.1)
test_lines=tolines(u'data//test_word.txt')


#导入数据
def pre_lines(lines):
    data=[]
    for lin in lines:
        word=[]
        tag=[]
        tmp=lin.split()
        for t in tmp:
            ind=t.index('/')
            word.append(t[:ind])
            tag.append(t[ind+1:])
        data.append((word,tag))
    return data

training_data=pre_lines(train_lines)#data('data//label-1400.txt')          
dev_data=pre_lines(dev_lines)#data('data')

def pre_test_lines(lines):
    data=[]
    for lin in lines:
        data.append(lin.split())
    return data
test_data=pre_test_lines(test_lines)

word_to_ix={}
for  sentence,tag in training_data:
    for word in sentence :
        if word not in word_to_ix:
            word_to_ix[word]=len(word_to_ix)
            
for  sentence,tag in dev_data:
    for word in sentence :
        if word not in word_to_ix:
            word_to_ix[word]=len(word_to_ix)
for  sentence in test_data:
    for word in sentence :
        if word not in word_to_ix:
            word_to_ix[word]=len(word_to_ix)
            








class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,tag_to_ix,embedding_dim,hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.tag_to_ix=tag_to_ix
        self.tagset_size=len(tag_to_ix)
        #vector table
        self.word_embeds=nn.Embedding(vocab_size,embedding_dim)
        #lstm模型
        self.lstm=nn.LSTM(embedding_dim,hidden_dim//2,
                          num_layers=1,bidirectional=True)
        #线性层,隐藏层到标签层
        self.hidden2tag=nn.Linear(hidden_dim,self.tagset_size)
        #参数,转移矩阵
        self.transitions=nn.Parameter(
            torch.randn(self.tagset_size,self.tagset_size))
        #初始化隐藏层
        self.hidden=self.init_hidden()
    #初始化隐藏层
    def init_hidden(self):
        return (autograd.Variable(torch.randn(2,1,self.hidden_dim//2)),
                autograd.Variable(torch.randn(2,1,self.hidden_dim//2)))
    
    def _forward_alg(self,feats):
        #do the forward algorithm to compute the partition function
        init_alphas=torch.Tensor(1,self.tagset_size).fill_(-10000.)
        
        init_alphas[0][self.tag_to_ix[START_TAG]]=0
        #wrap in a variable so that we will get automatic backprop
        forward_var=autograd.Variable(init_alphas)
        
        for feat in feats:
            alphas_t=[]
            for next_tag in range(self.tagset_size):
                emit_score=feat[next_tag].view(1,-1).expand(1,self.tagset_size)
                trans_score=self.transitions[next_tag].view(1,-1)
                next_tag_var=forward_var+trans_score+emit_score
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var=torch.cat(alphas_t).view(1,-1)
        terminal_var=forward_var+self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha=log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self,sentence):
        self.hidden=self.init_hidden()
        embeds=self.word_embeds(sentence).view(len(sentence),1,-1)
        lstm_out,self.hidden=self.lstm(embeds,self.hidden)
        lstm_out=lstm_out.view(len(sentence),self.hidden_dim)
        lstm_feats=self.hidden2tag(lstm_out)
        return  lstm_feats
    def _score_sentence(self,feats,tags):
        #gives the score of a provided tag sequence
        score=autograd.Variable(torch.Tensor([0]))
        tags=torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]),tags])
        for i, feat in enumerate(feats):
            score=score+\
                  self.transitions[tags[i+1],tags[i]]+feat[tags[i+1]]
        score=score+self.transitions[self.tag_to_ix[STOP_TAG],tags[-1]]
        return score
    
    def _viterbi_decode(self,feats):
        backpointers=[]
        #initialize the viterbi variables in lof space
        init_vvars=torch.Tensor(1,self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]]=0
        
        forward_var=autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t=[]
            viterbivars_t=[]
            for next_tag in range(self.tagset_size):
                next_tag_var=forward_var+self.transitions[next_tag]
                best_tag_id=argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            forward_var=(torch.cat(viterbivars_t)+feat).view(1,-1)
            backpointers.append(bptrs_t)
        terminal_var=forward_var+self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id=argmax(terminal_var)
        path_score=terminal_var[0][best_tag_id]
        
        #follow the back pointers to dcode the best path
        best_path=[best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id=bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start=best_path.pop()
        assert start==self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score,best_path
    
    def neg_log_likelihood(self,sentence,tags):
        feats=self._get_lstm_features(sentence)
        forward_score=self._forward_alg(feats)
        gold_score=self._score_sentence(feats,tags)
        return forward_score-gold_score
    
    def forward(self,sentence):
        lstm_feats=self._get_lstm_features(sentence)
        score,tag_seq=self._viterbi_decode(lstm_feats)
        return score,tag_seq
    
net=BiLSTM_CRF(len(word_to_ix),tag_to_ix,EMBEDDING_DIM,HIDDEN_DIM)
net.load_state_dict(torch.load('model1.pkl'))    
test_result=[]
for test_ in test_data:
    test_sent=prepare_sequence(test_,word_to_ix)
    test_tag=[ix_to_tag[ind] for ind in net.forward(test_sent)[1]]
    test_result.append((test_,test_tag))
        
with  open('test1.txt',"w") as f_wr:
    for test,tag,in test_result:
        f_wr.write((' '.join(test)+'###'+' '.join(tag)+'\n').encode('utf-8'))






