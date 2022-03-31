library(tidyverse)
library(readr)


#문자열 숫자로 전처리하기
docs = strsplit(rawdocs, split=' ') 
vocab = unique(unlist(docs))
for(i in 1:length(docs)) {
  docs[[i]] = match(docs[[i]], vocab)
  
}
words <- vector(length=length(vocab))

#하이퍼 파라미터 조정
K = 2
alpha = .1
beta = .001
iter.max = 2000

#문서-토픽, 토픽-단어 분포 함수
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
# 각 문서의 단어에 임의적으로 토픽 종류 할당
set.seed(323)
assign_topic = sapply(docs, function(x)
  sample(1: K, length(x), replace = TRUE))

#주제별 단어분포(word_topic) 초기화
topic_word = matrix(0, K, length(vocab))
topic_word = assign_word_topic(assign_topic, docs, topic_word)

document_topic = matrix(0, length(docs), K)  # 문서-토픽분포 초기화
document_topic = make_document_topic(assign_topic, K, document_topic)


#Procedure
#Gibbs sampling
for (iter in 1:iter.max){
  for (doc in 1:length(docs)){
    for (word in 1:length(docs[[doc]])){
      z_old = assign_topic[[doc]][word]  
      index = docs[[doc]][word]  
      
      # 두 분포에서 z 관련 정보를 제거 
      document_topic[doc, z_old] = document_topic[doc, z_old] - 1
      topic_word[z_old, index] = topic_word[z_old, index] - 1
      prob_topic = rep(0, K)
      for (k in 1:K){
        #P(Z|W) 특정단어가 특정 주제에 속할 확률 값
        prob_topic[k] = (document_topic[doc, k] + alpha)*
          (topic_word[k, w.index]+beta)/rowSums(topic_word+beta)[k]
      }
      
      #구한 확률을 이용해 새로운 주제(topic) 할당
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