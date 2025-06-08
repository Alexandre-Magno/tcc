import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape2(y: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean(np.abs((y - yhat) / y))


def mpe2(y: np.ndarray, yhat: np.ndarray) -> float:
    return np.mean((y - yhat) / y)


def ml_error(model: str, y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    mae = np.round(mean_absolute_error(y, yhat), 2)
    mape = np.round(mape2(y, yhat), 2)
    rmse = np.round(np.sqrt(mean_squared_error(y, yhat)), 2)
    mpe = np.round(mpe2(y, yhat), 2)

    return pd.DataFrame(
        {"Model Name": model, "MAE": mae, "MAPE": mape, "RMSE": rmse, "MPE": mpe},
        index=[0],
    )
