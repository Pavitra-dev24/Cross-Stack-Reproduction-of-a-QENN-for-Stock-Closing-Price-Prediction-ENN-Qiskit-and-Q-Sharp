import warnings, numpy as np
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

def persistence(d):
    return d['Xte'][:, -1].copy()

def arima_forecast(d, grid_p=(0,1,2,3), grid_q=(0,1,2)):
    """ARIMA(p,1,q); (p,q) picked by AIC on the training part only; one-step-ahead forecasts on the test part with parameters frozen."""
    s = d['scaled']; n_tr_end = d['n_train'] + 6       # training series = s[:n_tr_end]
    best = None
    for p in grid_p:
        for q in grid_q:
            try:
                r = ARIMA(s[:n_tr_end], order=(p, 1, q), trend='n').fit()
                if np.isfinite(r.aic) and (best is None or r.aic < best[0]): best = (r.aic, p, q, r)
            except Exception:
                pass
    aic, p, q, r = best
    full = r.apply(s)                                  # same parameters, filter over the entire series (no re-estimation)
    pred = full.get_prediction(start=n_tr_end, end=len(s) - 1, dynamic=False).predicted_mean
    return np.asarray(pred), (p, 1, q)
