import os
import numpy as np
import pandas as pd
import math
import csv

import transformers
from tqdm.notebook import trange, tqdm

from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import STSDataReader, TripletReader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryEmbeddingSimilarityEvaluator, SequentialEvaluator
from sentence_transformers.readers.InputExample import InputExample

import torch
from torch.utils.data import DataLoader, RandomSampler

from scipy.spatial.distance import cdist

torch.cuda.empty_cache()

my_model_path = 'msmarco/models/test_model5'

model_1 = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

dev_dataloader = torch.load(os.path.join('msmarco/models/test_model5', 'dev_dataloader.pth'))
train_dataloader =  torch.load(os.path.join('msmarco/models/test_model5', 'train_dataloader.pth'))

evaluator1 = BinaryEmbeddingSimilarityEvaluator(dev_dataloader)
evaluator2 = EmbeddingSimilarityEvaluator(dev_dataloader)
evaluator = SequentialEvaluator([evaluator1, evaluator2], main_score_function = lambda scores: scores[0])

optimizer_class = transformers.AdamW
optimizer_params = {'lr': 2e-6, 'eps': 1e-6, 'correct_bias': False}
train_loss = losses.CosineSimilarityLoss(model=model_1)

num_epochs = 100
warmup_steps = math.ceil(len(train_dataloader.dataset)*num_epochs / train_dataloader.batch_size*0.1) #10% of train data for warm-up

model_1.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          steps_per_epoch=1000,
          warmup_steps=warmup_steps,
          optimizer_class=optimizer_class,
          optimizer_params=optimizer_params,
          output_path=os.path.join(my_model_path, 'model_lre06_not_od')) # works only when you have an evaluator

model_1.save(os.path.join(my_model_path, 'model_lre06_not_od_final'))