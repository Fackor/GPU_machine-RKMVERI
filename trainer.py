#!/usr/bin/env python
# coding: utf-8

import nemo
import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.data.datasets import BertTextClassificationDataset
from nemo.collections.nlp.nm.data_layers.text_classification_datalayer import BertTextClassificationDataLayer
from nemo.collections.nlp.nm.trainables import SequenceClassifier

from nemo.collections.nlp.data.datasets import TextClassificationDataDesc

from nemo.backends.pytorch.common import CrossEntropyLossNM
from nemo.utils.lr_policies import get_lr_policy
from nemo.collections.nlp.callbacks.text_classification_callback import eval_iter_callback, eval_epochs_done_callback

import os
import json
import math
import numpy as np
import pandas as pd

import torch

from datetime import datetime

#defining hyperparameters and paths
DATA_DIR = input("Enter path to data directory : ")
if(DATA_DIR == ""):
    DATA_DIR="./data"

NUM_EPOCHS = input("Enter number of epochs to train : ")
if NUM_EPOCHS=="":
    NUM_EPOCHS=10
else:
    NUM_EPOCHS=int(NUM_EPOCHS)

BERT_CONFIG = input("Enter path to bert-config : ")
BERT_CHECKPOINT = input("Enter path to bert-checkpoint : ")
if(BERT_CONFIG == ""):
    BERT_CONFIG="../bert_uncased/bert-config.json"
if(BERT_CHECKPOINT == ""):
    BERT_CHECKPOINT="../bert_uncased/BERT-STEP-2285714.pt"
    
CLASS_BALANCING = bool(input("Enable weighted-loss class_balancing? [True/False] : "))

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M")
WORK_DIR = f'output_{dt_string}/'
    
AMP_OPTIMIZATION_LEVEL = 'O0'
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LEN = 64
BATCH_SIZE = 32 
    
NUM_GPUS = 1
LEARNING_RATE = 2e-5
OPTIMIZER = 'adam'
WEIGHT_DECAY = 0.01

FC_DROPOUT = 0.1
NUM_OUTPUT_LAYERS = 1

USE_CACHE = True

#create the neural module factory
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=None,
                                   optimization_level=AMP_OPTIMIZATION_LEVEL,
                                   log_dir=WORK_DIR,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True,)

#instantiate the pretrained bert model
bert = model = nemo_nlp.nm.trainables.get_pretrained_lm_model(pretrained_model_name=PRETRAINED_MODEL_NAME,
                                                              config=BERT_CONFIG,
                                                              vocab=None,
                                                              checkpoint=BERT_CHECKPOINT,)

#define the tokenizer
tokenizer = nemo.collections.nlp.data.tokenizers.get_tokenizer(tokenizer_name='nemobert',
                                                               pretrained_model_name=PRETRAINED_MODEL_NAME)

#get the hidden size from the bert model
hidden_size = bert.hidden_size

#add a data descriptor, this will automatically infer the 
#number of class labels and so on from the data
data_desc = TextClassificationDataDesc(data_dir=DATA_DIR, modes=["train", "dev"])

#add the loss according to the state of CLASS_BALANCING
if CLASS_BALANCING == True:
    print("You may need to increase the number of epochs for convergence.")
    loss = nemo.backends.pytorch.common.CrossEntropyLossNM(weight=data_desc.class_weights)
else:
    loss = CrossEntropyLossNM()

#now create the extra layer for inference from the bert model
mlp = SequenceClassifier(hidden_size=hidden_size,
                         num_classes=data_desc.num_labels,
                         dropout=FC_DROPOUT,
                         num_layers=NUM_OUTPUT_LAYERS,
                         log_softmax=False,)

#create the data layers
train_data = BertTextClassificationDataLayer(input_file=os.path.join(DATA_DIR, 'train.tsv'),
                                             tokenizer=tokenizer,
                                             max_seq_length=MAX_SEQ_LEN,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             use_cache=USE_CACHE)

val_data = BertTextClassificationDataLayer(input_file=os.path.join(DATA_DIR, 'dev.tsv'),
                                           tokenizer=tokenizer,
                                           max_seq_length=MAX_SEQ_LEN,
                                           batch_size=BATCH_SIZE,
                                           use_cache=USE_CACHE)

#get tokens from data layer
train_input, train_token_types, train_attn_mask, train_labels = train_data()
val_input, val_token_types, val_attn_mask, val_labels = val_data()

#generate BERT embeddings
train_embeddings = bert(input_ids=train_input,
                        token_type_ids=train_token_types,
                        attention_mask=train_attn_mask)
val_embeddings = bert(input_ids=val_input,
                      token_type_ids=val_token_types,
                      attention_mask=val_attn_mask)

#now create the training pipeline
train_logits = mlp(hidden_states=train_embeddings)
val_logits = mlp(hidden_states=val_embeddings)

train_loss = loss(logits=train_logits, labels=train_labels)
val_loss = loss(logits=val_logits, labels=val_labels)

train_data_size = len(train_data)
steps_per_epoch = math.ceil(train_data_size / (BATCH_SIZE * NUM_GPUS))

train_callback = nemo.core.SimpleLossLoggerCallback(tensors=[train_loss, train_logits],
                            print_func=lambda x:nemo.logging.info(f'Train loss: {str(np.round(x[0].item(), 3))}'),
                            tb_writer=nf.tb_writer,
                            get_tb_values=lambda x: [["train_loss", x[0]]],
                            step_freq=steps_per_epoch)

eval_callback = nemo.core.EvaluatorCallback(eval_tensors=[val_logits, val_labels],
                                            user_iter_callback=lambda x, y: eval_iter_callback(x, y, val_data),
                                            user_epochs_done_callback=lambda x:
                                                eval_epochs_done_callback(x, f'{nf.work_dir}/graphs'),
                                            tb_writer=nf.tb_writer,
                                            eval_step=steps_per_epoch)

ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                                             epoch_freq=1)

lr_policy_fn = get_lr_policy('WarmupAnnealing',
                             total_steps=NUM_EPOCHS * steps_per_epoch,
                             warmup_ratio=0.1)

#ask if training is to be done
TRAIN = bool(input("Do you want to train? [True|False]"))

if TRAIN:
	nf.train(tensors_to_optimize=[train_loss],
        	 callbacks=[train_callback, eval_callback, ckpt_callback],
         	 lr_policy=lr_policy_fn,
         	 optimizer=OPTIMIZER,
         	 optimization_params={"num_epochs": NUM_EPOCHS, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},)

#ask if inference file is to be created from the test set
INFERENCE = bool(input("Do you want to create the inference file?"))

if INFERENCE:
	#generate test layer
	test_data = BertTextClassificationDataLayer(input_file=os.path.join(DATA_DIR, 'test.tsv'),
        	                                    tokenizer=tokenizer,
                	                            max_seq_length=MAX_SEQ_LEN,
                        	                    batch_size=BATCH_SIZE)

	test_input, test_token_types, test_attn_mask, _ = test_data()

	test_embeddings = bert(input_ids=test_input,
        	                token_type_ids=test_token_types,
               		        attention_mask=test_attn_mask)

	#create the testing pipeline
	test_logits = mlp(hidden_states=test_embeddings)

	test_logits_tensors = nf.infer(tensors=[test_logits])

	test_probs = torch.nn.functional.softmax(torch.cat(test_logits_tensors[0]), dim=1).numpy()

	test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.tsv'), sep='\t')

	for i in range(data_desc.num_labels):
        	prob = test_probs[:, i]
        	test_df["prob{}".format(i)] = prob

	inference_file = os.path.join(DATA_DIR, 'inference', 'test_inference.tsv')

	test_df.drop(['label'], axis=1, inplace=True)

	test_df.to_csv(inference_file, sep='\t', index=False)

while(True):

	SINGLE_SENTENCE = bool(int("Classify a single sentence?"))
	if !SINGLE_SENTENCE:
		break
	
