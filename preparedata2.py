import csv
import re
import pickle
import time
import pandas as pd
from comet import Comet
import torch
import nltk
from sklearn.model_selection import train_test_split

relations=["xNeed","xWant","xAttr","xEffect",'xReact','oEffect','oReact','oWant']
word_pairs = {
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "you'd": "you would",
    "you're": "you are",
    "you'll": "you will",
    "i'm": "i am",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "couldn't": "could not",
    "i've": "i have",
    "we've": "we have",
    "can't": "cannot",
    "i'd": "i would",
    "i'd": "i would",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "there's": "there is",
    "there're": "there are",
}

def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

def get_commonsense(comet, item):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        # cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)
    return cs_list

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for i in range(len(s)):
        if i==0:
            str1+=s[i]
        else:
            str1=str1+" "+s[i]

    # return string  
    return str1 
   
# csv file name
filename = "Complaint data annotation (explain)_updated - cd.csv"
 
# initializing the titles and rows list
fields = []
rows = []
 
# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
     
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
 
    # get total number of rows
    print("Total no. of rows: %d"%(csvreader.line_num))
 
# printing the field names
print('Field names are:' + ', '.join(field for field in fields))
 
#  printing first 5 rows
print('\nFirst 5 rows are:\n')
cnt=0
data=[]
for row in rows:
    if len(row[1])>0:
        cnt+=1
        print(row[1])
        data.append(row)
        # text_data.append(row[0])
        # label_data.append(row[1])
print(cnt)
print(" ".join(data[0][1].split()))
print(data[0])
print(data[1])
print(data[2])


for i in range(len(data)):
    data[i][1]=" ".join(data[i][1].split())


print(data[0])

print(len(data))
print(data[0])
newdata=[]
for i in range(len(data)):
    newlist=[]
    print(data[i][1].split())
    for j in range(len(data[i][1].split())):
        if data[i][1].split()[j][0]=='@':
            continue
        else:
            newlist.append(data[i][1].split()[j])
    newdata.append(listToString(newlist))
    
print(newdata[0])

for i in range(len(data)):
    data[i][1]=newdata[i]

print(data[0][1])
print(data[1][1])

for i in range(len(data)):
    data[i][0]=re.sub(r'http\S+', '', data[i][0])

print(data[0])
print(data[1])
newdata=[]
for i in range(len(data)):
    newdata.append(data[i])
for i in range(len(data)):
    print(data[i])
    newdata[i]=data[i][1:3]
    newdata[i][2:]=data[i][-2:]

data=newdata

# data=data[:200]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

comet=Comet('COMEDMODELPATH',device)

sent=process_sent('i made a complaint in one of your stores 5 days ago and no one has dealt with it. whos best to contact at head office, please?')
cnt=0
newdata=[]
for i in range(len(data)):
    if data[i][1]=='1':
        data[i][1]='Complaint'
    else:
        data[i][1]='NonComplaint'
    data[i][1]=data[i][3]+' < '+data[i][2]+' > '+'< '+data[i][1]+' >'
    data[i]=data[i][:2]
    print(data[i])
    print(i)
    print(data[i][0])
    sent=data[i][0]
    sent=process_sent(sent)
    CS=get_commonsense(comet,sent)
    print('NEED: {}'.format(CS[0][0]))
    print('WANT: {}'.format(CS[1][0]))
    print('ATTR: {}'.format(CS[2][0]))
    print('EFFECT: {}'.format(CS[3][0]))
    print('REACT: {}'.format(CS[4][0]))
    print('oEffect: {}'.format(CS[5][0]))
    print('oreact: {}'.format(CS[6][0]))
    print('owant: {}'.format(CS[7][0]))
    data[i].append(CS[0][0]+' '+CS[1][0])
    print(data[i])
    input('ENTER')


df = pd.DataFrame(data, columns = ['text', 'label','CS'])
train_data,test_data = train_test_split(df,test_size=0.1, random_state=11)
train_data,eval_data = train_test_split(train_data,test_size=0.05, random_state=11)
print(train_data.head())
print(eval_data.head())
print(test_data.head())



