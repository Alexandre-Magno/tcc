import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime


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


def get_model_performance(models, x_train, y_train, x_test, y_test):

    # create dataframe
    df_performance = pd.DataFrame()

    for model in models:

        print("Training " + type(model).__name__ + " ...")

        # fit model
        model.fit(x_train, y_train)

        # prediction
        yhat = model.predict(x_test)

        # performance
        df = ml_error(type(model).__name__, y_test, yhat)

        # concat
        df_performance = pd.concat([df_performance, df])

        # reset index
        df_performance = df_performance.reset_index()

        # drops index
        df_performance.drop("index", axis=1, inplace=True)

    return df_performance, yhat


def models_cross_validation(models, x_training):

    df_performance = pd.DataFrame()

    for model in models:

        print("Training... " + type(model).__name__ + "...")
        mae = []
        mape = []
        rmse = []
        mpe = []
        for k in reversed(range(1, 6)):

            valid_start = x_training["data"].max() - pd.DateOffset(months=k * 6)
            valid_end = x_training["data"].max() - pd.DateOffset(months=(k - 1) * 6)

            print(
                "Iteration {} - Validating from {} to {}".format(
                    k, valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")
                )
            )
            # filtering dataset
            training = x_training[x_training["data"] < valid_start]
            validation = x_training[
                (x_training["data"] >= valid_start) & (x_training["data"] <= valid_end)
            ]

            # train and validation dataset
            # training
            xtrain = training.drop(["data", "valor"], axis=1)
            ytrain = training["valor"]

            # validation
            xvalid = validation.drop(["data", "valor"], axis=1)
            yvalid = validation["valor"]

            # model
            model.fit(xtrain, ytrain)

            # predict
            yhat = model.predict(xvalid)

            # performance
            df = ml_error(
                (type(model).__name__ + "  KFold: {}".format(k)),
                yvalid,
                yhat,
            )

            mae.append(df["MAE"])
            mape.append(df["MAPE"])
            rmse.append(df["RMSE"])
            mpe.append(df["MPE"])
            # concat
            df_performance = pd.concat([df_performance, df])

            # reset index
            df_performance = df_performance.reset_index()

            # drops index
            df_performance.drop("index", axis=1, inplace=True)

        mae_cv = (
            np.round(np.mean(mae), 2).astype(str)
            + " +/- "
            + np.round(np.std(mae), 2).astype(str)
        )
        mape_cv = (
            np.round(np.mean(mape), 2).astype(str)
            + " +/- "
            + np.round(np.std(mape), 2).astype(str)
        )
        rmse_cv = (
            np.round(np.mean(rmse), 2).astype(str)
            + " +/- "
            + np.round(np.std(rmse), 2).astype(str)
        )

        mpe_cv = (
            np.round(np.mean(mpe), 2).astype(str)
            + " +/- "
            + np.round(np.std(mpe), 2).astype(str)
        )

        df_cv = pd.DataFrame(
            {
                "Model Name": (type(model).__name__ + " - Cross-Validation "),
                "MAE": mae_cv,
                "MAPE": mape_cv,
                "RMSE": rmse_cv,
                "MPE": mpe_cv,
            },
            index=[0],
        )

        # concat
        df_performance = pd.concat([df_performance, df_cv])

        # reset index
        df_performance = df_performance.reset_index()

        # drops index
        df_performance.drop("index", axis=1, inplace=True)

    return df_performance
