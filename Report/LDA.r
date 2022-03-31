library(tidyverse)
library(readr)


#���ڿ� ���ڷ� ��ó���ϱ�
docs = strsplit(rawdocs, split=' ') 
vocab = unique(unlist(docs))
for(i in 1:length(docs)) {
  docs[[i]] = match(docs[[i]], vocab)
  
}
words <- vector(length=length(vocab))

#������ �Ķ���� ����
K = 2
alpha = .1
beta = .001
iter.max = 2000

#����-����, ����-�ܾ� ���� �Լ�
make_document_topic = function(assign_topic, K, document_topic) {
  document_topic = document_topic * 0
  for (doc in 1:length(assign_topic)){
    for (topic in 1:K){
      document_topic[doc, topic] = sum(assign_topic[[doc]] == topic)
    }
  }
  return (document_topic)
}

assign_word_topic = function(assign_topic, docs, topic_word) {
  topic_word = topic_word * 0
  for (doc in 1:length(docs)){
    for (word in 1:length(docs[[doc]])){
      topic_index = assign_topic[[doc]][word]
      word_index = docs[[doc]][word]
      topic_word[topic_index, word_index] = topic_word[topic_index, word_index] + 1    
    }
  }
  return (topic_word)
}

#Initialization
# �� ������ �ܾ ���������� ���� ���� �Ҵ�
set.seed(323)
assign_topic = sapply(docs, function(x)
  sample(1: K, length(x), replace = TRUE))

#������ �ܾ����(word_topic) �ʱ�ȭ
topic_word = matrix(0, K, length(vocab))
topic_word = assign_word_topic(assign_topic, docs, topic_word)

document_topic = matrix(0, length(docs), K)  # ����-���Ⱥ��� �ʱ�ȭ
document_topic = make_document_topic(assign_topic, K, document_topic)


#Procedure
#Gibbs sampling
for (iter in 1:iter.max){
  for (doc in 1:length(docs)){
    for (word in 1:length(docs[[doc]])){
      z_old = assign_topic[[doc]][word]  
      index = docs[[doc]][word]  
      
      # �� �������� z ���� ������ ���� 
      document_topic[doc, z_old] = document_topic[doc, z_old] - 1
      topic_word[z_old, index] = topic_word[z_old, index] - 1
      prob_topic = rep(0, K)
      for (k in 1:K){
        #P(Z|W) Ư���ܾ Ư�� ������ ���� Ȯ�� ��
        prob_topic[k] = (document_topic[doc, k] + alpha)*
          (topic_word[k, w.index]+beta)/rowSums(topic_word+beta)[k]
      }
      
      #���� Ȯ���� �̿��� ���ο� ����(topic) �Ҵ�
      z_new = sample(1:K, 1, prob = prob_topic) 
      if(z_new != z_old) assign_topic[[doc]][word] = z_new
      
      topic_word = assign_word_topic(assign_topic, docs, topic_word)
      document_topic = make_document_topic(assign_topic, K, document_topic)
    }
  }
}


##For compariosn, utilize LDA package
library(lda)

K <- 5
num.iterations <- 10000
corpus <- lexicalize(docs, lower=TRUE)
set.seed(323)
result <- lda.collapsed.gibbs.sampler(corpus$documents, K, 
                                      corpus$vocab,num.iterations, 0.01, 0.001, 
                                      compute.log.likelihood = TRUE)
topics<- top.topic.words(result$topics, 10, by.score = TRUE)