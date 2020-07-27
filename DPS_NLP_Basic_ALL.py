# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:29:03 2020

@author: Dipin Singh
"""

##TOPIC SCOVERED -
1. POS 
2. Chunking
3. Chinking


import nltk
nltk.download('tagsets')

print(nltk.help.upenn_tagset())


## 1
Sentence = nltk.word_tokenize("The grass is always green on the other side")
print(nltk.pos_tag(Sentence))


##2
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer()
tokenizer = custom_sent_tokenizer.tokenize(train_text)

corpus=[]
for i in tokenizer:
    words=nltk.word_tokenize(i)
    tagged=nltk.pos_tag(words)
    corpus.append(tagged)
    
    
    
    
##########################################################
## Chunking
    

words = nltk.word_tokenize("The grass is always green on the other side")
tagged = nltk.pos_tag(words)


for word in words:
    filterset = r""" chunkWOrds: { <NN.?>*<JJ>} """
    chunkparser1 = nltk.RegexpParser(filterset)
    chunked=chunkparser1.parse(tagged)
  #  chunked.draw()
    
print(chunked)



##CHINKING

words = nltk.word_tokenize("The grass is always green on the other side")
tagged = nltk.pos_tag(words)


for word in words:
    filterset = r""" chunkWOrds: { <.*>}
                                } <NN>{"""
    chunkparser1 = nltk.RegexpParser(filterset)
    chunked=chunkparser1.parse(tagged)
    
print(chunked)



## Nammed ENTITY REGONIZATION

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer()
tokenizer = custom_sent_tokenizer.tokenize(train_text)

for i in tokenizer:
    words=nltk.word_tokenize(i)
    tagged=nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged)
    namedEnt.draw()

