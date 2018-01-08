#encoding:utf-8
#导入神经网络模型所需的模块
import torch 
import torch.nn as nn
import torch.nn.function as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):
	def __init__(self,args):
		super(BiLSTM_CRF,self).__init__()
		self.embedding_dim=args.embedding_dim
		self.hidden_dim=args.hidden_dim
		self.vocab_size=args.vocab_size
		self.tag_to_ix=args.tag_to_ix
		self.tagset_size=len(args.tag_to_ix)
		self.word_embeds=nn.Embedding(self.vocab_size,self.embedding_dim)
		self.lstm=nn.LSTM(self.embedding_dim,hidden_dim//2,
							num_layers=1,bidirectional=True)
		self.hidden2tag=nn.Linear(hidden_dim,self.tagset_size)
		self.transitions=nn.Parameter(
			torch.randn(self.tagset_size,self.tagset_size))
		self.hidden=self.init_hidden()
	def forward(self,sentence):
		#隐藏输入
		self.hidden=(autograd.Variable(torch.randn(2,1,self.hidden_dim//2)),
					autograd.Variable(torch.randn(2,1,self.hidden_dim//2)))
		#将句子转换为句子矩阵，即词语变成词向量
		embeds=self.word_embeds(sentence).view(len(sentence),1,-1)
		#输入lstm,得到输出，其中输出大小为（句子长度，隐藏层*2）
		lstm_out,self.hidden=self.lstm(embeds,self.hidden)
		#转换lstm输出，（句子长度，隐藏层*2）
		lstm_out=lstm_out.view(len(sentence),self.hidden_dim)
		#全连接层，得到（句子长度，tagset_size）
		lstm_feats=self.hidden2tag(lstm_out)
		backpointers=[]
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
		best_path=[best_tag_id]
		
		for bptrs_t in reversed(backpointers):
			best_tag_id=bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		start=best_path.pop()
		assert start==self.tag_to_ix[START_TAG]
		best_path.reverse()
		
		
		



