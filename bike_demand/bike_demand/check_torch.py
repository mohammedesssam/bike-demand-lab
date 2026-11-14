"""Reload the saved network and independently reproduce its held-out MAE."""
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error
from bike_demand.features import make_features
from bike_demand.train import ROOT

torch.set_num_threads(1)
c=torch.load(ROOT/'artifacts/mlp.pt',map_location='cpu',weights_only=True)
net=nn.Sequential(nn.Linear(c['input_dim'],24),nn.ReLU(),nn.Linear(24,1))
net.load_state_dict(c['state_dict']); net.eval()
d=pd.read_csv(ROOT/'data/day.csv');d=d[d.dteday>='2012-10-01']
x=joblib.load(ROOT/'artifacts/torch_preprocessor.joblib').transform(make_features(d))
with torch.no_grad(): p=net(torch.tensor(x,dtype=torch.float32)).numpy().ravel()*c['target_std']+c['target_mean']
mae=float(mean_absolute_error(d.cnt,np.maximum(p,0)))
expected=json.loads((ROOT/'reports/metrics.json').read_text())['test']['pytorch_mlp']['mae']
assert abs(mae-expected)<0.01,(mae,expected)
print(f'Saved PyTorch checkpoint reload passed; held-out MAE = {mae:.6f}')
