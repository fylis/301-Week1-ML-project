import numpy
import pandas
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# --- Réglage du degré du polynôme ---
POLY_DEGREE = 3

FEATURES = [
    "surf_hab",
    "qualite_materiau",
    "surface_sous_sol",
    "n_garage_voitures",
    "n_pieces",
]


def data_preprocessing(df: pandas.DataFrame) -> (numpy.array, numpy.array):
    y = df["prix"].values
    x = df[FEATURES].values
    return x, y


def model_fit(X: numpy.array, Y: numpy.array):
    # model = Pipeline(steps=[
    #     ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
    #     ("linreg", LinearRegression())
    # ])
    # model.fit(X, Y)
    # return model
    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    cv=5,
                    n_alphas=150,
                    random_state=0,
                    max_iter=50000,
                ),
            ),
        ]
    )
    model.fit(X, Y)
    return model


def model_predict(model, X: numpy.array) -> numpy.array:
    return model.predict(X)
