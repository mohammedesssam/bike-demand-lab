import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from bike_demand.api import app, PredictionRequest, predict
from bike_demand.features import FEATURES, make_features
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

PAYLOAD={'day':'2012-10-15','temperature_normalized':0.45,'humidity_percent':65,
         'windspeed_normalized':0.18,'weather':1,'holiday':False}

@pytest.fixture
def client():
    with TestClient(app) as c: yield c

def test_data_hash():
    assert hashlib.sha256((ROOT/'data/day.csv').read_bytes()).hexdigest() == 'a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be'

def test_no_target_features():
    d=pd.read_csv(ROOT/'data/day.csv'); X=make_features(d)
    assert list(X.columns)==FEATURES
    assert not set(X.columns)&{'cnt','casual','registered','instant'}
    assert np.isfinite(X.to_numpy()).all()

def test_chronology():
    m=json.loads((ROOT/'reports/metrics.json').read_text())['splits']
    assert m['train']['to'] < m['validation']['from'] < m['test']['from']
    assert m['validation']['to'] < m['test']['from']
    assert sum(v['rows'] for v in m.values()) == 731

def test_health_and_ui(client):
    assert client.get('/health').json()['status']=='ok'
    assert 'Bike Demand Lab' in client.get('/').text
    assert client.get('/docs').status_code==200

def test_prediction(client):
    a=client.post('/predict',json=PAYLOAD)
    assert a.status_code==200 and a.json()['predicted_daily_rentals']>=0
    assert a.json()==client.post('/predict',json=PAYLOAD).json()

@pytest.mark.parametrize('field,value', [('day','2026-11-14'),('day','2010-12-31'),
    ('humidity_percent',101),('temperature_normalized',-1),('windspeed_normalized',1.1),('weather',4)])
def test_invalid(client,field,value):
    assert client.post('/predict',json={**PAYLOAD,field:value}).status_code==422

def test_reject_extra_and_missing(client):
    assert client.post('/predict',json={**PAYLOAD,'cnt':99999}).status_code==422
    assert client.post('/predict',json={}).status_code==422

def test_offline_online_parity(client):
    import joblib
    d=pd.read_csv(ROOT/'data/day.csv').iloc[650]
    body={'day':d.dteday,'temperature_normalized':d.temp,'humidity_percent':d.hum*100,
          'windspeed_normalized':d.windspeed,'weather':int(d.weathersit),'holiday':bool(d.holiday)}
    model=joblib.load(ROOT/'artifacts/model.joblib')['pipeline']
    expected=round(float(max(model.predict(make_features(pd.DataFrame([d])))[0],0)))
    assert client.post('/predict',json=body).json()['predicted_daily_rentals']==expected

def test_metric_recomputation():
    from sklearn.metrics import mean_absolute_error
    d=pd.read_csv(ROOT/'reports/test_predictions.csv')
    m=json.loads((ROOT/'reports/metrics.json').read_text())
    assert mean_absolute_error(d.actual,d.prediction)==pytest.approx(m['test']['selected_model']['mae'])
