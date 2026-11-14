"""One feature contract shared by offline training and online prediction."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['temp', 'hum', 'windspeed', 'trend', 'month_sin', 'month_cos']
CATEGORICAL = ['weekday', 'holiday', 'weathersit']
FEATURES = NUMERIC + CATEGORICAL

def make_features(raw):
    d = pd.to_datetime(raw['dteday'])
    return pd.DataFrame({
        'temp': raw['temp'].astype(float),
        'hum': raw['hum'].astype(float),
        'windspeed': raw['windspeed'].astype(float),
        'trend': (d - pd.Timestamp('2011-01-01')).dt.days / 365.25,
        'month_sin': np.sin(2*np.pi*(d.dt.month-1)/12),
        'month_cos': np.cos(2*np.pi*(d.dt.month-1)/12),
        'weekday': d.dt.dayofweek,
        'holiday': raw['holiday'].astype(int),
        'weathersit': raw['weathersit'].astype(int),
    }, index=raw.index)[FEATURES]

def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])
