#building a chatbot

#import libraries
import numpy as np
import tensorflow as tf
import re
import time

#..........Data Preprocessing.......

#import datasets.....
lines=open('movie_lines.txt', encoding='utf-8' ,errors='ignore' ).read().split('\n')
conv=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

#creating dictionaries.....to map each line with its id
id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
    
#creating list.....of conversations id
id2conv=[]
for conver in conv[:-1]:
    _conver=conver.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    id2conv.append(_conver.split(','))
    
#seperating question and answers
questions=[]
answers=[]
for converse in id2conv:
    for i in range(len(converse)-1):
        questions.append(id2line[converse[i]])
        answers.append(id2line[converse[i+1]])
    
#Cleaning all text 
def cleantext(text):
    text=text.lower() #lowercase
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't","would not",text)
    text=re.sub(r"can't","can not",text)
    text=re.sub(r"[-()\"/@;:.{}+=~|?,]","",text)
    return text
    
#cleaning questions
clean_questions=[]
for quest in questions:
    clean_questions.append(cleantext(quest))
#cleaning answers    
clean_answers=[]
for ans in answers:
    clean_answers.append(cleantext(ans))
    
#creating dictonaries map with number of occurences
word2count={}
for quest in clean_questions:
    for word in quest.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
for ans in clean_answers:
    for word in ans.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
        
#creationg 2 dictonaries question to map and answer to map with unique integer [tokenization]
threshold=10
questionword2int={}
word_number=0
for word,count in word2count.items():
    if count>=threshold:
        questionword2int[word]=word_number
        word_number+=1
answerword2int={}
word_number=0
for word,count in word2count.items():
    if count>=threshold:
        answerword2int[word]=word_number
        word_number+=1   

#last tokens to the previous dictonaries
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionword2int[token]=len(questionword2int)+1
    answerword2int[token]=len(answerword2int)+1
#create inverse dictionary of answerword2int dictionary
answerint2word={w_i:w for w,w_i in answerword2int.items()}

#adding EndOfString to end of each of answer
for i in range(len(clean_answers)):
    clean_answers[i]+=' <EOS>'

#translate question and answer by integers
question2int=[]
for quest in clean_questions:
    ints=[]
    for word in quest.split():
        if word not in questionword2int:
            ints.append(questionword2int['<OUT>'])
        else:
            ints.append(questionword2int[word])
    question2int.append(ints)
answer2int=[]
for ans in clean_answers:
    ints=[]
    for word in ans.split():
        if word not in answerword2int:
            ints.append(answerword2int['<OUT>'])
        else:
            ints.append(answerword2int[word])
    answer2int.append(ints)
#sorting questions and answers by length 
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,25 + 1):
    for i in enumerate(question2int):
        if len(i[1])==length:
            sorted_clean_questions.append(question2int[i[0]])
            sorted_clean_answers.append(answer2int[i[0]])
