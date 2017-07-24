#_*_ coding:utf-8_*_

#Weibo sentiment analysis based on machine learning
from __future__ import print_function
from  gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import  sys
reload(sys)
import  numpy as np
from  sklearn.preprocessing import scale
from  sklearn import svm

#Read the data from datafile
with open("pos.txt", "r") as pos_input:
    pos_weibo = pos_input.readlines()

with open("neg.txt", "r") as neg_input:
    neg_weibo = neg_input.readlines()

#Labled each sentence(1 for positive and 0 for negtive)and split them to train_data and test_data
y =np.concatenate((np.ones(len(pos_weibo)), np.zeros(len(neg_weibo))))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_weibo, neg_weibo)), y, test_size=0.2)

# tfidf testing
def testPipline():  
  
    text_clf = Pipeline([('vect', CountVectorizer()),   
                ('tfidf', TfidfTransformer()),   
                ('clf', svm.SVC(kernel='rbf',verbose=True)),   
                ])  
    text_clf.fit(x_train, y_train)  
         
    predicted = text_clf.predict(x_test)  
      
    accuracy=np.mean(predicted == y_test)  
    #print accuracy   
    print ("The accuracy of test is %s" %accuracy)  
      
    print(classification_report(y_test, predicted))  
	
testPipline()



