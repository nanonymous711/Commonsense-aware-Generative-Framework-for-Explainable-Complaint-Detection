import os
import time
import argparse
import numpy as np

import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
import nltk
from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer
from transformers.modeling_bart import make_padding_mask

from classifier.textcnn import TextCNN
from utils.optim import ScheduledOptim
from utils.helper import optimize, evaluate
from utils.helper import cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator
import pickle
import time
# from comet import Comet

import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
relations=["xNeed","xWant"]
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# comet=Comet('/DATA/sriparna/BartRLCM/pre-trained-formality-transfer/comet-atomic_2020_BART/',device)


# text='The new <USER> stinks 10mins to take my order and another 15 to get it . And stop asking my name like we’re friends'
text='Real disappointed in <USER> leaving me high and dry. Ordered some new Iowa gear Tues with 1 day shipping and it hasnt even shipped yet.'
# text='Hey guys, I love this product featured on today but don’t see a price? Help a girl out? <  to see the price > <  to ask for a price >'
sent=process_sent(text)
# CS=get_commonsense(comet,sent)
# print(CS[0][0])
# print(CS[1][0])
# text=text+' < '+CS[0][0]+' > '+ '< '+CS[1][0]+' >'
# print(text)
torch.cuda.set_device(3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# for token in ['<E>', '<F>']:
#     tokenizer.add_tokens(token)
# if opt.dataset == 'em':
#     domain = tokenizer.encode('<E>', add_special_tokens=False)[0]
# else:
#     domain = tokenizer.encode('<F>', add_special_tokens=False)[0]
model = BartModel.from_pretrained("facebook/bart-large")
model.config.output_past = True
model = BartForMaskedLM.from_pretrained("facebook/bart-large",
                                        config=model.config)
# model.to(device).eval()
model.to('cuda').eval()
directory=os.listdir('checkpointsbaseAndSSAndSpanwithCS')
for filename in directory:
		print(filename)
		if filename!='38000.chkpt':
			continue
		# if filename!='38000.chkpt':
		# 	continue
		print(torch.cuda.current_device())



		model.load_state_dict(torch.load('checkpointsbaseAndSSAndSpanwithCS/'+filename,map_location='cuda:3'))
		print('loaded')



		preds=[]

		src=tokenizer.encode(text,return_tensors='pt')
		generated_ids=model.generate(src.to(device),num_beams=5,max_length=30)
		text=[tokenizer.decode(g,skip_special_tokens=True,clean_up_tokenization_spaces=False) for g in generated_ids][0]
		print(text)
		print(text)
		preds.append(text)
