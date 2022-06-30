
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import os
import time
import argparse

import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter

!pip --q install tensorboardX

from google.colab import drive
drive.mount('/content/drive')

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m/u.data"

MAIN_PATH = '/content/drive/MyDrive/CF_PROJECT/'

DATA_PATH = MAIN_PATH + '/data/ml-1m/steam-200k.csv'

MODEL_PATH = MAIN_PATH + 'models/'

import random
import numpy as np 
import pandas as pd 
import torch


import os
import random
import numpy as np 
import torch
import json

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True






#metrics
def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def metrics(model, test_loader, top_k, device):
	HR, NDCG, MAP = [], [], []

	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)
  

		predictions = model(user, item)
  
		# print(predictions) # Sigmoid outputs
  
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()
	
		# print(recommends) # List of topk recommendations

		ng_item = item[0].item() # leave one-out evaluation has only one item per user, the first item is positive, rest negatives
		HR.append(hit(ng_item, recommends))
		NDCG.append(ndcg(ng_item, recommends))
		MAP.append(apk([ng_item], recommends, top_k))

	return np.mean(HR), np.mean(NDCG), np.mean(MAP)







device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
# seed_everything(seed)

def process(ratings, user_col="user_id", item_col="item_id", ratings_col="rating", n_users=5000, n_items=5000, encode=False):
    """
    If explicit, converts to implicit. Keeps only those users that occur at least twice.
    Adds column timestamp if not present and renames the columns as desired.
    """
    if "timestamp" not in ratings.columns:
        ratings["timestamp"] = np.random.randint(10000, 99999, size=ratings.shape[0])
    ratings = ratings.rename(columns={user_col: "user_id", item_col:"item_id", ratings_col:"rating"})
    ratings = ratings.loc[:, ["user_id", "item_id", "rating", "timestamp"]]
    # ratings = ratings[ratings["rating"]>=1.]
    ratings["rating"] = 1.0

    if encode:
        unique_items = ratings['item_id'].unique()
        dict = {item: i for i, item in enumerate(unique_items)}
        ratings['item_id'] = ratings['item_id'].apply(lambda x: dict[x])

    v = ratings['item_id'].value_counts()[:n_items]
    ratings = ratings[ratings['item_id'].isin(v.index)]

    v = ratings['user_id'].value_counts()[:n_users]
    ratings = ratings[ratings['user_id'].isin(v.index)]

    ratings = ratings.groupby('user_id').filter(lambda x: x['user_id'].shape[0] > 1)

    return ratings

ml_1m = pd.read_csv(DATA_PATH, names=['user_id', 'item_id', 'type', 'rating', '_'])
print(ml_1m.head(), end="\n\n")


ml_1m = process(ml_1m, encode=True, n_users=9000, n_items=9000)





#parameters
seed = 42
lr = 5e-4
dropout = 0.1
batch_size = 256
epochs = 20
top_k = 10
factor_num = 32
layers = [64,32,16,8]
num_ng = 4
num_ng_test = 100
out = True

params= {
    'seed':seed,
    'lr':lr,
    'dropout': dropout,
    'batch_size': batch_size,
    'top_k': top_k,
    'factor_num': factor_num,
    'layers' : layers,
    'num_ng' : num_ng,
    'num_ng_test' : num_ng_test,
    'out' : out,
    'bilinear1': True,
    'bilinear2': True
}

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

print(num_users)
print(num_items)

# construct the train and test datasets
data = NCF_Data(params, ml_1m)
train_loader =data.get_train_instance()
test_loader =data.get_test_instance()

params1 = params.copy()
params1['bilinear1'] = False
params1['bilinear2'] = False
params1['name'] = "wo_bl"

params2 = params.copy()
params2['bilinear1'] = True
params2['bilinear2'] = False
params2['name'] = "wo_bl2"

params3 = params.copy()
params3['bilinear1'] = False
params3['bilinear2'] = True
params3['name'] = "wo_bl1"

params4 = params.copy()
params4['bilinear1'] = True
params4['bilinear2'] = True
params4['name'] = "w_bl"

params5 = params.copy()
params5['bilinear1'] = False
params5['bilinear2'] = False
params5['layers'] = [62, 128, 64, 32, 16]
params5['name'] = "wo_bl_deep"

params6 = params.copy()
params6['bilinear1'] = True
params6['bilinear2'] = True
params6['name'] = "w_bl_concat"

parameters = [params1]

# set model and loss, optimizer
for i, params in enumerate(parameters):
    print(f"[INFO] Starting with parameters{i+1}!")
    print(params)
    model = NeuMF(params, num_users, num_items)
    model = model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    logger = {
        "HR@10": [],
        "NDCG@10": [],
        "MAP@10": []
    }

    MODEL = f'Steam_Neu_MF_{params["name"]}'
    MODEL_DIR = os.path.join(MODEL_PATH, MODEL)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # train, evaluation
    best_hr = 0
    for epoch in range(1, epochs+1):
        model.train() # Enable dropout (if have).
        start_time = time.time()

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG, MAP = metrics(model, test_loader, top_k, device)
        logger["HR@10"].append(HR)
        logger["NDCG@10"].append(NDCG)
        logger["MAP@10"].append(MAP)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}\t MAP: {}".format(np.mean(HR), np.mean(NDCG), MAP))

        if HR > best_hr:
            best_hr, best_ndcg, best_map, best_epoch = HR, NDCG, MAP, epoch
            if out:
                torch.save(model, os.path.join(MODEL_DIR, 'model.pth'))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}, MAP = {:.3f}".format(
                                        best_epoch, best_hr, best_ndcg, best_map))
    with open(os.path.join(MODEL_DIR, 'logs.json'), "w") as f:
        json.dump(logger, f)

