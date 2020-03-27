import os
import numpy as np
import pandas as pd
import math
import csv

import transformers
from tqdm.notebook import trange, tqdm

from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import STSDataReader, TripletReader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryEmbeddingSimilarityEvaluator, SequentialEvaluator, TripletEvaluator
from sentence_transformers.readers.InputExample import InputExample

import torch
from torch.utils.data import DataLoader, RandomSampler

from scipy.spatial.distance import cdist

torch.cuda.empty_cache()

my_model_path = '/run/media/root/Windows/Users/agnes/Downloads/data/msmarco/train_results/test_wiki'

model_wiki = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

dev_dataloader = torch.load(os.path.join(my_model_path, 'dev_dataloader.pth'))
train_dataloader =  torch.load(os.path.join(my_model_path, 'train_dataloader.pth'))

evaluator = TripletEvaluator(dev_dataloader)

optimizer_class = transformers.AdamW
optimizer_params = {'lr': 2e-4, 'eps': 1e-6, 'correct_bias': False}
train_loss = losses.TripletLoss(model=model_wiki)

num_epochs = 4
warmup_steps = math.ceil(len(train_dataloader.dataset)*num_epochs / train_dataloader.batch_size*0.05) #5% of train data for warm-up

model_wiki.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          steps_per_epoch=8000,
          warmup_steps=warmup_steps,
          optimizer_class=optimizer_class,
          optimizer_params=optimizer_params,
          output_path=os.path.join(my_model_path, 'model_2')) # works only when you have an evaluator

model_1.save(os.path.join(my_model_path, 'model_2_final'))