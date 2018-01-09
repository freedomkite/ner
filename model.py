#encoding:utf-8
#导入神经网络模型所需的模块
import torch 
import torch.nn as nn
import torch.nn.function as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(1)
		
class BiLSTM_CRF(nn.Module):
    def __init__(self,args,vocab_size,tag_to_ix,embedding_dim,hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim=args.embedding_dim
        self.hidden_dim=args.hidden_dim
        self.vocab_size=args.vocab_size
        self.tag_to_ix=args.tag_to_ix
        self.tagset_size=len(args.tag_to_ix)
		#词向量
        self.word_embeds=nn.Embedding(args.vocab_size,args.embedding_dim)
        #lstm模型
        self.lstm=nn.LSTM(args.embedding_dim,args.hidden_dim//2,
                          num_layers=1,bidirectional=True)
        #线性层,隐藏层到标签层
        self.hidden2tag=nn.Linear(self.hidden_dim,self.tagset_size)
		
        #参数,转移矩阵,用于vitebi
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
		#隐层输入
        self.hidden=self.init_hidden()
		#词向量输入
        embeds=self.word_embeds(sentence).view(len(sentence),1,-1)
		#lstm输入，双向
        lstm_out,self.hidden=self.lstm(embeds,self.hidden)
		#调整输出形状
        lstm_out=lstm_out.view(len(sentence),self.hidden_dim)
		#经过全连接层
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
        
        #follow the back pointers to decode the best path
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
		#经过bilstm层，得到矩阵（len(sentence),self.tagset_size）
        lstm_feats=self._get_lstm_features(sentence)
		#给予viterbi算法进行解码，可以获得最佳解码路径以及分数
        score,tag_seq=self._viterbi_decode(lstm_feats)
        return score,tag_seq
		
		
		



