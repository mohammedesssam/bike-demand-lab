"""Local/API deployment. Never load an untrusted pickle/joblib model."""
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from bike_demand.features import make_features

ROOT = Path(__file__).resolve().parents[1]
state = {}

@asynccontextmanager
async def lifespan(app):
    state['model'] = joblib.load(ROOT/'artifacts/model.joblib')
    state['metrics'] = json.loads((ROOT/'reports/metrics.json').read_text())
    yield
    state.clear()

app = FastAPI(title='Bike Demand Lab', version='1.0.0', lifespan=lifespan)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    day: date = Field(ge=date(2011,1,1), le=date(2012,12,31))
    temperature_normalized: float = Field(ge=0, le=1, description='UCI daily temp value, not Celsius')
    humidity_percent: float = Field(ge=0, le=100)
    windspeed_normalized: float = Field(ge=0, le=1, description='UCI windspeed normalized by 67')
    weather: Literal[1,2,3]
    holiday: bool = False

@app.get('/')
def index(): return FileResponse(ROOT/'static/index.html')

@app.get('/health')
def health(): return {'status':'ok', 'model':state['model']['model_name']}

@app.get('/metrics')
def metrics(): return state['metrics']

@app.post('/predict')
def predict(payload: PredictionRequest):
    raw = pd.DataFrame([{'dteday':payload.day.isoformat(), 'temp':payload.temperature_normalized,
        'hum':payload.humidity_percent/100, 'windspeed':payload.windspeed_normalized,
        'weathersit':payload.weather, 'holiday':int(payload.holiday)}])
    value = float(np.maximum(state['model']['pipeline'].predict(make_features(raw))[0], 0))
    return {'predicted_daily_rentals':round(value), 'model':state['model']['model_name'],
        'scope':'Historical Washington DC 2011-2012 educational demo. Not validated for current dates, other cities, or operational use.',
        'warning':'Weather is an input scenario. This is not a live forecast or an uncertainty interval.'}
