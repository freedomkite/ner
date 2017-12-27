#encoding:utf-8
#读取文件
from collections import Counter
import numpy as np
def readFile(src):
	buff=[]
	with open(src,'r',encoding='utf-8') as f_r:
		for line in f_r:
			line=line.strip()
			if line:
				buff.append(line)
	return buff
	
#处理每一行
def proLine(line):
	tmp=line.split()
	word=[]
	pos=[]
	label=[]
	for w in tmp:
		ind1=w.index('/')   #求取斜杠的索引
		ind2=w.index('_')   #求取下划线的索引
		word.append(w[:ind1])
		pos.append(w[inde1+1:ind2])
		label.append(w[ind2+1:])
	return word,pos,label
	
#导入数据
def loadData(src):
	sentences=[]
	poses=[]
	labels=[]
	for line in readFile(src):
		tmp=proLine(line)
		sentences.append(tmp[0])
		poses.append(tmp[1])
		labels.append(tmp[2])
	return sentences,poses,labels
	
#构建词语字典
def build_dict(sentences):
	word_count=Counter()
	max_len=0
	word_count['unk']+=1
	for sent in sentences:
		if len(sent)>max_len:
			max_len=len(sent)
		for w in sent:
			word_count[w]+=1
	ls=word_count.most_common()
	word_dict={w[0]:index for (index,w) in enumerate(ls)}
	return word_dict,max_len
	
#将句子进行向量化
def vectorize(data,word_dict,pos_dict,label_dict,max_len):
	sentences,poses,labels=data
	num_data=len(sentences)
	sent_vec=np.zeros((num_data,max_len),dtype=int)
	pos_vec=np.zeros((num_data,max_len),dtype=int)
	label_vec=np.zeros((num_data,max_len),dtype=int)
	for idx,(sent,pos,label) in enumerate(zip(sentences,poses,labels)):
		if len(sent)>max_len:
			sent=sent[:max_len]
		vec=[word_dict[w] if w  in word_dict else 0 for w in sent]
		sent_vec[idx,:]=vec
		
		if len(pos)>max_len:
			pos=pos[:max_len]
			
		vec1=[pos_dict[p] if p in pos_dict else 0 for p in pos]
		pos_vec[idx,:]=vec1
		
		if len(labels)>max_len:
			label=label[:max_len]
		vec2=[label_dict[la] if la in label_dict else 0 for la in label]
		
	return sent_vec,pos_vec,label_vec
	


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

		
	