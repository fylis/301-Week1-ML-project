import numpy
import pandas
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = [
    "surf_hab",
]


def data_preprocessing(df: pandas.DataFrame) -> (numpy.array, numpy.array):
    y = df["prix"]
    x = df[FEATURES].values
    return x, y


def model_fit(X: numpy.array, Y: numpy.array):
    # model = LinearRegression()
    # model.fit(X, Y)
    # return model
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    cv=5,  # 5-fold CV
                    n_alphas=100,  # search grid
                    random_state=0,
                    max_iter=20000,
                ),
            ),
        ]
    )
    model.fit(X, Y)
    return model


def model_predict(model, X: numpy.array) -> numpy.array:
    return model.predict(X)
