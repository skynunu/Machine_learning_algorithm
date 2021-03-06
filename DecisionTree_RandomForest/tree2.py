#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function 


# In[2]:


# 구글 드라이브 연결
from google.colab import drive
drive.mount('/content/drive')


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


# 드라이브 sys.path에 폴더 경로 추가
import sys
sys.path.append('/content/drive/MyDrive/mechine_deep_learning/DecisionTree_RandomForest')


# In[5]:


import random
import numpy as np 
import pandas as pd 
import collections
from collections import Counter
import tree1
from tree1 import DecisionTree


# In[6]:


class RandomForest(object):
  def __init__(self, numberOfTrees=20, max_depth= 10, min_samples_split = 2, min_gain = 1e-4, random_state=100):
     self.forests = [] # 트리들을 저장할 변수
     self.numberOfTrees = numberOfTrees # 트리 개수
     self.max_depth = max_depth
     self.min_samples_split = min_samples_split
     self.min_gain= min_gain
     self.random_state = random_state

  # bootstrap sampling
  def getBag(self, x_data, y_data, random_state) : 

      # random_num에 따라 random_seed의 범위 설정
      random_seed = random.randint(1,random_state)
      random.seed(random_seed)
      
      data_list = range(0,len(y_data)) # 데이터 개수 만큼의 범위 리스트 설정
      idx= []

      #데이터 개수 만큼 sampling 해서 데이터 만들기
      for i in range(0, len(y_data)):
        idx.append(random.choice(data_list))
        bagged_x = pd.DataFrame(x_data.iloc[idx,:])
        bagged_y = pd.Series(y_data.iloc[idx])
      return bagged_x, bagged_y

  def fit(self,x_data, y_data) :
    
    #numberOfTrees만큼 트리를 만들고 샘플링 된 데이터를 학습시키키
    for i in range(self.numberOfTrees) :
      dtree = DecisionTree(self.max_depth, self.min_samples_split, self.min_gain, rf = True)
      bagged_x, bagged_y = self.getBag(x_data, y_data, self.random_state)
      dtree.fit(bagged_x, bagged_y)
      self.forests.append(dtree)


  def classify(self,test_data) :
    result= []  
    # test_data를 하나의 row씩 읽어서 결과값 예측하기
    for _, row in test_data.iterrows() : 
      votes = []
      for forest in self.forests :
        votes += forest.classify(row)
      votes_counts = Counter(votes)
      # most vote를 answer로 정하기
      answer = votes_counts.most_common(1)[0][0]
      result.append(answer)
    return result

    


# In[ ]:




