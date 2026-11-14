"""Reproducible chronological model selection, held-out evaluation and artifacts."""
import hashlib
import json
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bike_demand.features import make_features, preprocessor, FEATURES

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

def metrics(y, p):
    p = np.maximum(p, 0)
    return {'mae': float(mean_absolute_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'r2': float(r2_score(y, p))}

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data_path = ROOT/'data/day.csv'
    d = pd.read_csv(data_path).sort_values('dteday').reset_index(drop=True)
    assert len(d) == 731 and d['dteday'].is_unique
    assert not d.isna().any().any()
    assert (d['casual'] + d['registered']).equals(d['cnt'])
    X = make_features(d); y = d['cnt']
    tr = d['dteday'] <= '2012-06-30'
    va = d['dteday'].between('2012-07-01', '2012-09-30')
    te = d['dteday'] >= '2012-10-01'
    assert not set(FEATURES) & {'cnt', 'casual', 'registered', 'instant'}
    candidates = {f'ridge_alpha_{a}': Ridge(alpha=a) for a in [1., 10., 100.]}
    for leaves in [7, 15]:
        candidates[f'gradient_boosting_{leaves}_leaves'] = HistGradientBoostingRegressor(
            max_leaf_nodes=leaves, max_iter=150, learning_rate=.05,
            l2_regularization=10., random_state=SEED)
    val_scores = {}; fitted = {}
    for name, estimator in candidates.items():
        p = Pipeline([('preprocess', preprocessor()), ('model', estimator)])
        p.fit(X[tr], y[tr]); fitted[name] = p
        val_scores[name] = metrics(y[va], p.predict(X[va]))
    # A small, actually trained PyTorch comparison. Validation chooses its checkpoint.
    prep = preprocessor(); a = prep.fit_transform(X[tr]); b = prep.transform(X[va])
    ty = float(y[tr].mean()); sy = float(y[tr].std())
    net = nn.Sequential(nn.Linear(a.shape[1], 24), nn.ReLU(), nn.Linear(24, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=.001)
    xt = torch.tensor(a, dtype=torch.float32)
    yt = torch.tensor(((y[tr]-ty)/sy).to_numpy(), dtype=torch.float32).reshape(-1, 1)
    xv = torch.tensor(b, dtype=torch.float32)
    best = float('inf'); checkpoint = None; best_epoch = None
    for epoch in range(600):
        net.train(); optimizer.zero_grad()
        loss = nn.functional.mse_loss(net(xt), yt); loss.backward(); optimizer.step()
        net.eval()
        with torch.no_grad(): pv = net(xv).numpy().ravel()*sy+ty
        error = mean_absolute_error(y[va], np.maximum(pv, 0))
        if error < best:
            best = float(error); best_epoch = epoch+1
            checkpoint = {k: v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(checkpoint); net.eval()
    with torch.no_grad():
        vpred = net(xv).numpy().ravel()*sy+ty
        tpred = net(torch.tensor(prep.transform(X[te]), dtype=torch.float32)).numpy().ravel()*sy+ty
    val_scores['pytorch_mlp'] = metrics(y[va], vpred)
    # Serving uses the best scikit-learn model by validation MAE; the separate MLP
    # comparison remains reported, not misrepresented as the deployed estimator.
    selected = min(fitted, key=lambda n: val_scores[n]['mae'])
    model = fitted[selected]  # freeze the train-only fit; no test refitting or tuning
    pred = np.maximum(model.predict(X[te]), 0)
    # Train-only weekday means, no future outcome or lag leakage.
    weekday_mean = y[tr].groupby(X.loc[tr,'weekday']).mean()
    baseline = X.loc[te,'weekday'].map(weekday_mean).to_numpy()
    result = {
        'seed': SEED, 'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'rows': len(d), 'features': FEATURES, 'excluded_leakage_columns': ['cnt','casual','registered','instant'],
        'splits': {name:{'rows':int(mask.sum()),'from':d.loc[mask,'dteday'].min(),
                    'to':d.loc[mask,'dteday'].max()} for name, mask in [('train',tr),('validation',va),('test',te)]},
        'selection_rule':'Lowest validation MAE among scikit-learn candidates; held-out test not used in selection.',
        'validation': val_scores, 'selected_model':selected,
        'test': {'selected_model':metrics(y[te], pred), 'weekday_mean_baseline':metrics(y[te],baseline),
                 'pytorch_mlp':metrics(y[te],tpred)},
        'pytorch': {'best_epoch':best_epoch, 'epochs_tried':600, 'hidden_units':24,
                    'checkpoint_selection':'validation MAE', 'target_standardization':'train-only mean and standard deviation'},
        'scope':'Historical Washington DC 2011-2012 weather-conditional daily rental-count demo, not production demand forecasting.'
    }
    (ROOT/'artifacts').mkdir(exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True)
    joblib.dump({'pipeline':model, 'model_name':selected, 'features':FEATURES}, ROOT/'artifacts/model.joblib')
    joblib.dump(prep, ROOT/'artifacts/torch_preprocessor.joblib')
    torch.save({'state_dict':checkpoint,'input_dim':a.shape[1], 'target_mean':ty,'target_std':sy}, ROOT/'artifacts/mlp.pt')
    (ROOT/'reports/metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    pd.DataFrame({'date':d.loc[te,'dteday'], 'actual':y[te], 'prediction':pred,
                  'baseline':baseline}).to_csv(ROOT/'reports/test_predictions.csv',index=False)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11,4.5))
    ax.plot(pd.to_datetime(d.loc[te,'dteday']), y[te], label='Actual', color='#102f52', lw=1.7)
    ax.plot(pd.to_datetime(d.loc[te,'dteday']), pred, label='Selected model', color='#13a894',lw=1.5)
    ax.set(ylabel='Recorded rentals per day', title='Untouched chronological test: October-December 2012')
    ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(ROOT/'reports/test_predictions.png', dpi=150)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
