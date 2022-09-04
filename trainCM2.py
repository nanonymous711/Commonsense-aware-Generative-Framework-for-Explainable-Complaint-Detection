
import os
import time
import argparse
import numpy as np

import torch
from torch import cuda
from torch.nn import CrossEntropyLoss
from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer
from transformers.modeling_bart import make_padding_mask
from transformers import BartTokenizer
# from transformers.models.bart.modeling_bart import make_padding_mask
from utils.helper import make_padding_mask
from classifier.textcnn import TextCNN
from utils.optim import ScheduledOptim
from utils.helper import optimize, evaluate
from utils.helper import cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator
import pickle
import time
import random

# device = 'cuda' if cuda.is_available() else 'cpu'
torch.cuda.set_device(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
	tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


	base = BartModel.from_pretrained("facebook/bart-base")
	model = BartForMaskedLM.from_pretrained('facebook/bart-base', config=base.config)
	# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
	# base = MultimodalBartModel.from_pretrained("facebook/bart-base")
	# model = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base', config=base.config)
	model.to(device).train()


	with open("traintextCMwithCS", "rb") as fp:
	    traintext = pickle.load(fp)
	with open("trainCMlabelAndSSAndSpan", "rb") as fp:
	    trainlabels = pickle.load(fp)


	print(len(traintext))
	print(traintext[0])
	print(trainlabels[0])
	# time.sleep(120)
	for i in range(len(traintext)):
		traintext[i]=str(traintext[i])
	for i in range(len(trainlabels)):
		trainlabels[i]=str(trainlabels[i])
	# time.sleep(120)

	with open("validtextCMwithCS", "rb") as fp:
	    validtext = pickle.load(fp)
	with open("validCMlabelAndSSAndSpan", "rb") as fp:
	    validlabels = pickle.load(fp)

	for i in range(len(validtext)):
		validtext[i]=str(validtext[i])
	for i in range(len(validlabels)):
		validlabels[i]=str(validlabels[i])


	print(len(validtext))



	trainsrc_seq, traintgt_seq = [], []
	max_len=30

	# for k in range(len(traintext)):
	f1 = traintext
	f2 = trainlabels
	index = [i for i in range(len(f1))]
	random.shuffle(index)
	index = index[:int(len(index) * 1.0)]
	for i, (s, t) in enumerate(zip(f1, f2)):
	    if i in index:
	    	print(s)
	    	if len(s)==0:
	    		continue
	    	s = tokenizer.encode(s)
	    	t = tokenizer.encode(t)
	    	s = s[:min(len(s) - 1, max_len)] + s[-1:]
	    	t = t[:min(len(t) - 1, max_len)] + t[-1:]
	    	trainsrc_seq.append(s)
	    	traintgt_seq.append([tokenizer.bos_token_id]+t)


	validsrc_seq, validtgt_seq = [], []
	max_len=30
	# for k in range(len(validtext)):
	f1 = validtext
	f2 = validlabels
	index = [i for i in range(len(f1))]
	random.shuffle(index)
	index = index[:int(len(index) * 1.0)]
	for i, (s, t) in enumerate(zip(f1, f2)):
	    if i in index:
	        s = tokenizer.encode(s)
	        t = tokenizer.encode(t)
	        s = s[:min(len(s) - 1, max_len)] + s[-1:]
	        t = t[:min(len(t) - 1, max_len)] + t[-1:]
	        # s[0] = domain
	        validsrc_seq.append(s)
	        validtgt_seq.append([tokenizer.bos_token_id]+t)

	train_loader, valid_loader = BARTIterator(trainsrc_seq, traintgt_seq,
	                                          validsrc_seq, validtgt_seq)

	print('done')


	loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
	optimizer = ScheduledOptim(torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
	                     betas=(0.9, 0.98), eps=1e-09), 1e-5, 10000)



	tab = 0
	eval_loss = 1e8
	total_loss_ce = []
	# total_loss_sc = []
	total_loss_co = []
	start = time.time()
	train_iter = iter(train_loader)
	for step in range(1, 60002):
		print('current {}'.format(step))

		try:
		    batch = next(train_iter)
		except:
		    train_iter = iter(train_loader)
		    batch = next(train_iter)

		src, tgt = map(lambda x: x.to(device), batch)
		src_mask = make_padding_mask(src, tokenizer.pad_token_id)
		src_mask = 1 - src_mask.long() if src_mask is not None else None
		logits = model(src, attention_mask=src_mask, decoder_input_ids=tgt)[0]
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = tgt[..., 1:].contiguous()
		loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
		                  shift_labels.view(-1))
		total_loss_ce.append(loss_ce.item())

		loss_sc, loss_co = torch.tensor(0), torch.tensor(0)
		# if opt.sc and (200 < step or len(train_loader) < step):
		#     idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
		#     loss_sc = cal_sc_loss(logits, idx, cls, tokenizer, opt.style)
		#     total_loss_sc.append(loss_sc.item())
		if (10000 < step or len(train_loader)< step):
			# print('RL')
			idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
			loss_co = cal_bl_loss(logits, tgt, idx, tokenizer)
			total_loss_co.append(loss_co.item())

		optimize(optimizer, loss_ce + loss_co)

		if step % 100 == 0:
		    lr = optimizer._optimizer.param_groups[0]['lr']
		    print('[Info] steps {:05d} | loss_ce {:.4f} | '
		          'loss_co {:.4f} | lr {:.6f} | second {:.2f}'.format(
		        step, np.mean(total_loss_ce),
		        np.mean(total_loss_co), lr, time.time() - start))
		    total_loss_ce = []
		    # total_loss_sc = []
		    total_loss_co = []
		    start = time.time()


		if step%2000==0:
			torch.save(model.state_dict(), 'newbartcheckpointsbasebaseAndSSAndSpanwithCS/{}.chkpt'.format(
			        step))

		# if ((len(train_loader) > 200
		#      and step % 200 == 0)
		#         or (len(train_loader) < 200
		#             and step % len(train_loader) == 0)):
		#     valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn,
		#                                      tokenizer, step)
		    # if eval_loss >= valid_loss:
		    #     torch.save(model.state_dict(), 'checkpoints/{}_{}_{}_{}.chkpt'.format(
		    #         opt.model, opt.dataset, opt.order, opt.style))
		#         print('[Info] The checkpoint file has been updated.')
		#         eval_loss = valid_loss
		#         tab = 0
		#     else:
		#         tab += 1
		#     if tab == opt.patience:
		#         exit()





if __name__ == "__main__":
    main()