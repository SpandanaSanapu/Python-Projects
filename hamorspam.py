# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:14:56 2017

@author: saikr
"""

#import os
import pandas as pd
import numpy as np
import re
import regex
import emoji
from collections import Counter
import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from Tkinter import *
from tkinter import *
#import scikitplot as skplt
import matplotlib.pyplot as plt
#import tkinter.simpledialog
#import tkinter.messagebox 


# Function to count the number of letters in each text including spaces
def count_letters(txt):  
    l = (list(filter(lambda x: x not in "", txt)))
    return len(l)

#Function that Counts the number of numeric strings in each text and they are strictly numeric 
#(eg: [my27r2, 00,9449876685] gives a count of 2 instead of 3)
def Numeric_Count(txt):
    words = txt.split()
    #print("Words",words)
    mynewlist = [s for s in words if s.isdigit()]
    #print("my new list",mynewlist)
    return len(mynewlist)

#Function that counts how many times a word has repeated in each text and gives the value of most repeated word
def most_freq_word(txt):
    words = re.findall(r'\w+',txt)
    conv_to_lower = [word.lower() for word in words]
    count_word = Counter(conv_to_lower)
    new = dict(count_word)
    #print(new)
    try:
        v, k = max((v, k) for k, v in new.items())
    except:
        v, k = 0, 0
    return v

#Function to count number of currency symbols in each text
def currency_count(txt):
    words = regex.findall(r'\p{Sc}',txt)
    #print(words)
    return len(words)


        
#Function to convert dataframe to Arff file   
def p2d(data_frame, target_file, relation='default'):
    pandas_iter = iter(data_frame.iterrows())
    arff.dump(target_file,[list(r[1]) for r in pandas_iter],relation=relation,names=data_frame.columns)


#Function to calculate accuracy score. The parameters to this function are the computed confusion matrix values from weka.    
#def Accuracy_score(TP,TN,FP,FN):
#    return (TP+TN)/(TP+TN+FP+FN)

    
#Function begins here:    
def main():

    
    data = pd.read_excel('C:\\Users\\saikr\\Desktop\\Trust\\spamdetect.xlsx',encoding = 'utf-8') 
    data = data.replace(['ham','spam'],[0, 1]) 
    data['Char_Count']=0
    data['Numeric_Count']=0
    data['most_freq_word']=0
    data['currency_count']=0
    #data['emoji_count']=0
    for i in np.arange(0,len(data.Text)):
        data.loc[i,'Char_Count'] = count_letters(data.loc[i,'Text'])
        data.loc[i,'Numeric_Count'] = Numeric_Count(data.loc[i,'Text'])
        data.loc[i,'most_freq_word'] = most_freq_word(data.loc[i,'Text'])
        data.loc[i,'currency_count'] = currency_count(data.loc[i,'Text']) 
        #data.loc[i,'emoji_count'] = emoji_count(data.loc[i,'Text']) 
    data2 = data[['Char_Count','Numeric_Count','most_freq_word','currency_count','Class']]
    
    #Reads data frame to CSV File
    data2.to_csv('C:\\Users\\saikr\\Desktop\\Trust\\spam4.csv', sep=',',index=False)
    
    #Converting dataframe to arff file
    #p2d(data2,'final_output.arff',relation='spam_detection')

        
    
    
    data = pd.read_csv('C:\\Users\\saikr\\Desktop\\Trust\\output_csv_file.csv',encoding = 'utf-8')
    X = data.drop('Class',axis=1)
    y = data['Class']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    X_train.shape, y_train.shape
    X_test.shape, y_test.shape
    
    #print("X_train,Y_train",X_train,y_train)
    
    
 # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier(n_estimators=100)
    
    model = lr
    model.fit(X_train,y_train)
    #print("Printing Model",y_train)
    pred = model.predict(X_test)
    print("Predicted values",pred)
    print("Classification Report",metrics.classification_report(y_test, pred))
    print("Confusion Matrix",metrics.confusion_matrix(y_test, pred))
    print("Accuracy:",accuracy_score(y_test, pred))


# #############################################################################
# Plot calibration plots

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:
                # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()
    
    Actual_0 = 0
    Actual_1 = 0
    for i in y_test:
        if(i==0):
            Actual_0 +=1
        else:
            Actual_1 +=1

    pred_0 = 0
    pred_1 = 0
    for i in pred:
        if(i==0):
            pred_0 +=1
        else:
            pred_1 +=1
            
    #kplt.metrics.plot_roc_curve(y_test, pred)
    #plt.show()
    
    print("Actual 0's in y",Actual_0)
    print("Actual 1's in y",Actual_1)
    print("predicted 0's in y",pred_0)
    print("predicted 1's in y",pred_1)
    
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
    
    #final_actuals = pd.Series(y_test, name='Actual')
    #final_pred = pd.Series(pred, name='Predicted')
    df_confusion = pd.crosstab(y_test, pred)
    
    print("check confusion Matrix:",df_confusion)
    def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
        plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(df_confusion.columns))
        plt.xticks(tick_marks, df_confusion.columns, rotation=45)
        plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)

    plot_confusion_matrix(df_confusion)
    

        
    messg = input("Enter a message to check ham or spam : ")
    #print(messg)
    
    letcount = count_letters(messg)
    Numcount = Numeric_Count(messg)
    mfw = most_freq_word(messg)
    currcount = currency_count(messg)
        #data.loc[i,'emoji_count'] = emoji_count(data.loc[i,'Text']) 
    data2 = [letcount,Numcount,mfw,currcount]
    print(data2)
    make_pred = model.predict(data2)
    print(make_pred)
    make_pred.shape
    
    #print(make_pred)
    if make_pred==0:
        print("Message entered is ham")
    else:
        print("Message entered is spam")
        
            
    


    
   
    
    
    #Reads data frame to CSV File
    #data2.to_csv('C:\\Users\\saikr\\Desktop\\Trust\\spam4.csv', sep=',',index=False)
    
    #Converting dataframe to arff file
    #p2d(data2,'final_output.arff',relation='spam_detection')



if __name__ == "__main__":
    main()
    
   
    
    