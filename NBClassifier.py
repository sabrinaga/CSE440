# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:11:27 2019

@author: Sabrina Garcia
"""
import os
import nltk
import math


######################################################################
# TRAINING
######################################################################
K = 0.1 #parameter alpha/pseudo-count 
POS_PATH = "C:\\Users\\Sabrina Garcia\\OneDrive\\Documents\\CSE440\\trainData\\posTrain\\"
NEG_PATH = "C:\\Users\\Sabrina Garcia\\OneDrive\\Documents\\CSE440\\trainData\\negTrain\\"
POS_TEST_PATH = "C:\\Users\\Sabrina Garcia\\OneDrive\\Documents\\CSE440\\TestData (1)\\posTest\\"
NEG_TEST_PATH = "C:\\Users\\Sabrina Garcia\\OneDrive\\Documents\\CSE440\\TestData (1)\\negTest\\"

posFileList = [POS_PATH+f for f in os.listdir(POS_PATH)]
negFileList = [NEG_PATH+f for f in os.listdir(NEG_PATH)]
                       
posTestList = [POS_TEST_PATH+f for f in os.listdir(POS_TEST_PATH)]
negTestList = [NEG_TEST_PATH+f for f in os.listdir(NEG_TEST_PATH)]               
                             
cleanText = []   
for fileName in posFileList:
    with  open(fileName,mode='r',encoding='ISO-8859-1') as f:
        rawText = f.read()
                    
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')
    cleanText += tokenizer.tokenize(rawText)

    
cleanText2 = []
for fileName2 in negFileList:
    with  open(fileName2,mode='r',encoding='ISO-8859-1') as f:
        rawText2 = f.read()
    tokenizer2 = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')
    cleanText2 += tokenizer2.tokenize(rawText2)
                
#Dict containg a count of how many times it appears in either Pos or Neg TrainFile
wordCountPos = {}
wordCountNeg = {}

#Dictionaries containing a word and the conditional probability 
#that said word is either positive or negative.
conditionalProbPos = {}
conditionalProbNeg = {}


#update the dictionaries containing the pos word and word count
for word in cleanText:
    if word not in wordCountPos:
        wordCountPos[word] = 1
    else:
        wordCountPos[word] = wordCountPos[word] + 1 

#sum all the values in the positive word count dict
posValueSum = 0 
for word in wordCountPos:
    posValueSum = posValueSum + wordCountPos[word] 

#update the dictionaries containing the neg word and word count
for word in cleanText2:
    if word not in wordCountNeg:
        wordCountNeg[word] = 1
    else:
        wordCountNeg[word] = wordCountNeg[word] + 1 

#sum all the values in the negative word count dict
negValueSum = 0
for word in wordCountNeg:
    negValueSum = negValueSum + wordCountNeg[word] 


a = [] #List of words which appear in Pos
b = [] #List of words which appear in Neg 
for key in wordCountPos.keys():
    a.append(key)    
for key in wordCountNeg.keys():
    b.append(key)

#d is the num of unique words in both neg and pos documents
d = len(set(a+b))

#in the following code we do addative/laplace smoothing
for word in wordCountPos:
    conditionalProbPos[word] = (wordCountPos[word] + K) / (posValueSum + K*d)
    if word not in wordCountNeg:
        conditionalProbNeg[word] =  K / (negValueSum + K*d)       
    
for word in wordCountNeg:
    conditionalProbNeg[word] = (wordCountNeg[word] + K) / (negValueSum + K*d)
    if word not in wordCountPos:
        conditionalProbPos[word] =  K / (negValueSum + K*d)
    

######################################################################
# TESTING
######################################################################
resultsP = {} #for the files in /posTest/ gives True if file was labeled correctly and false otherwise
resultsN = {} #for the files in /negTest/ gives True if file was labeled correctly and false otherwise


#For the file in \posTest\ open the file, clean it and calculate the total probabilities and
#make a comparison to verify weather it was labeled correctly

for fileName3 in posTestList:
    with  open(fileName3,mode='r',encoding='ISO-8859-1') as f:
        rawText3 = f.read()
    tokenizer3 = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')
    cleanText3 = tokenizer3.tokenize(rawText3) #cleantext is list of words
    totalProbPos = 0  #totalProbPos and totalProbNeg are total probability pos and total prob neg
    totalProbNeg = 0
    for word in cleanText3:
        if word in conditionalProbPos:
            totalProbPos += math.log(conditionalProbPos[word]) #sub probability multiplication with addition by using log space
        if word in conditionalProbNeg:
            totalProbNeg += math.log(conditionalProbNeg[word])
    if totalProbPos > totalProbNeg: 
        resultsP[fileName3]= True
        
    if totalProbNeg > totalProbPos:
        resultsP[fileName3] = False
        
            
            
            
#For the file in \negTest\ open the file, clean it and calculate the total probabilities and
#make a comparison to verify weather it was labeled correctly            
            
for fileName4 in negTestList:
    with  open(fileName4,mode='r',encoding='ISO-8859-1') as f:
        rawText4 = f.read()
    tokenizer4 = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+[\']*[a-zA-Z]+|[;!?$]')
    cleanText4 = tokenizer4.tokenize(rawText4) #cleantext is list of words
    totalProbPos1 = 0
    totalProbNeg2 = 0
    for word in cleanText4:
        if word in conditionalProbPos:
            totalProbPos1 += math.log(conditionalProbPos[word]) #sub probability multiplication with addition by using log space
        if word in conditionalProbNeg:
            totalProbNeg2 += math.log(conditionalProbNeg[word])
    if totalProbPos1 > totalProbNeg2:
        resultsN[fileName4]= False
        
    if totalProbNeg2 > totalProbPos1:
        resultsN[fileName4] = True
                   

                    
            
TP = sum(resultsP.values())
FN = len(resultsP.values()) - TP
TN = sum(resultsN.values())
FP = len(resultsN.values()) - TN
acc = (TP+TN)/(TP+FN+FP+TN)
print('Pred \ Gold \t P \t N')
print('P \t \t '+str(TP)+' \t '+str(FP))
print('N \t \t '+str(FN)+' \t '+str(TN))



acc = (TP+TN)/float(TP+FN+FP+TN)
print('accuracy is: ')
print(acc)            