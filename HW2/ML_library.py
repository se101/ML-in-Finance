import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math as mth

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit



def ls_sharpe_characteristic(df, char_col, date_col="yyyymm", ret_col="ret"):

    tmp = df[[date_col, ret_col, char_col]].dropna().copy()

    # Percentile ranking dentro de cada mes
    tmp["rank"] = tmp.groupby(date_col)[char_col].rank(pct=True)

    # Top 10% y Bottom 10%
    top = tmp[tmp["rank"] >= 0.9]
    bottom = tmp[tmp["rank"] <= 0.1]

    # Long-short mensual
    ls = (
        top.groupby(date_col)[ret_col].mean()
        - bottom.groupby(date_col)[ret_col].mean()
    )

    return float(np.sqrt(12) * ls.mean() / ls.std(ddof=1))

def get_OOS(series_y_test, series_y_train, series_y_hat):
    y_test  = np.asarray(series_y_test).reshape(-1)
    y_train = np.asarray(series_y_train).reshape(-1)
    y_hat   = np.asarray(series_y_hat).reshape(-1)

    sse_model = np.sum((y_test - y_hat) ** 2)
    y_bar = np.mean(y_train)
    sse_bench = np.sum((y_test - y_bar) ** 2)

    return 1 - sse_model / sse_bench


def linear_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos):

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

    y_hat_test = pipe.predict(X_test.to_numpy())
    r_oos_test = get_OOS(y_test, y_train, y_hat_test)

    X_train_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_train_full = np.hstack([
        np.asarray(y_train).reshape(-1),
        np.asarray(y_test).reshape(-1)
    ])

    pipe.fit(X_train_full, y_train_full)

    y_hat_oos = pipe.predict(X_oos.to_numpy()).reshape(-1)
    r_oos_oos = get_OOS(y_oos, y_train_full, y_hat_oos)

    return ["linear", float(r_oos_test), float(r_oos_oos)], y_hat_oos


def ridge_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos, alphas=None):

    if alphas is None:
        alphas = np.logspace(-4, 4, 30)

    best_alpha = None
    best_r2 = -np.inf

    for a in alphas:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=a))
        ])

        pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

        y_hat_test = pipe.predict(X_test.to_numpy())
        r_oos_test = get_OOS(y_test, y_train, y_hat_test)

        if r_oos_test > best_r2:
            best_r2 = r_oos_test
            best_alpha = a

    X_train_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_train_full = np.hstack([
        np.asarray(y_train).reshape(-1),
        np.asarray(y_test).reshape(-1)
    ])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=best_alpha))
    ])

    final_pipe.fit(X_train_full, y_train_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r_oos_oos = get_OOS(y_oos, y_train_full, y_hat_oos)

    return ["ridge", float(best_r2), float(r_oos_oos)], y_hat_oos


def lasso_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos, alphas=None):

    if alphas is None:
        alphas = np.logspace(-4, 1, 30)

    best_alpha = None
    best_r2 = -np.inf

    for a in alphas:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=a, max_iter=20000))
        ])

        pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

        y_hat_test = pipe.predict(X_test.to_numpy())
        r_oos_test = get_OOS(y_test, y_train, y_hat_test)

        if r_oos_test > best_r2:
            best_r2 = r_oos_test
            best_alpha = a

    X_train_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_train_full = np.hstack([
        np.asarray(y_train).reshape(-1),
        np.asarray(y_test).reshape(-1)
    ])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=best_alpha, max_iter=20000))
    ])

    final_pipe.fit(X_train_full, y_train_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r_oos_oos = get_OOS(y_oos, y_train_full, y_hat_oos)

    return ["lasso", float(best_r2), float(r_oos_oos)], y_hat_oos


def elasticnet_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos,
                              alphas=None, l1_ratios=None):

    if alphas is None:
        alphas = np.logspace(-4, 1, 20)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_alpha = None
    best_l1 = None
    best_r2 = -np.inf

    for a in alphas:
        for l1 in l1_ratios:

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000))
            ])

            pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

            y_hat_test = pipe.predict(X_test.to_numpy())
            r_oos_test = get_OOS(y_test, y_train, y_hat_test)

            if r_oos_test > best_r2:
                best_r2 = r_oos_test
                best_alpha = a
                best_l1 = l1

    X_train_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_train_full = np.hstack([
        np.asarray(y_train).reshape(-1),
        np.asarray(y_test).reshape(-1)
    ])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=20000))
    ])

    final_pipe.fit(X_train_full, y_train_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r_oos_oos = get_OOS(y_oos, y_train_full, y_hat_oos)

    return ["elastic_net", float(best_r2), float(r_oos_oos)], y_hat_oos


def all_linear_models_fit(X_train, y_train, X_test, y_test, X_oos, y_oos):

    results = []
    yhat_dict = {}

    info, yhat = linear_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos)
    results.append(info); yhat_dict[info[0]] = yhat

    info, yhat = ridge_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos)
    results.append(info); yhat_dict[info[0]] = yhat

    info, yhat = lasso_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos)
    results.append(info); yhat_dict[info[0]] = yhat

    info, yhat = elasticnet_regression_fit(X_train, y_train, X_test, y_test, X_oos, y_oos)
    results.append(info); yhat_dict[info[0]] = yhat

    return results, yhat_dict


def rbf_ridge_fit(X_train, y_train, X_test, y_test, X_oos, y_oos,
                  alphas=None, gamma=1.0, n_components=500):

    if alphas is None:
        alphas = np.logspace(-4, 4, 20)

    best_alpha = None
    best_r2 = -np.inf

    for a in alphas:

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=42)),
            ("model", Ridge(alpha=a))
        ])

        pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

        y_hat_test = pipe.predict(X_test.to_numpy())
        r2 = get_OOS(y_test, y_train, y_hat_test)

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = a

    X_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_full = np.hstack([np.asarray(y_train).reshape(-1),
                        np.asarray(y_test).reshape(-1)])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=42)),
        ("model", Ridge(alpha=best_alpha))
    ])

    final_pipe.fit(X_full, y_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r2_oos = get_OOS(y_oos, y_full, y_hat_oos)

    return ["rbf_ridge", float(best_r2), float(r2_oos)], y_hat_oos


def rbf_lasso_fit(X_train, y_train, X_test, y_test, X_oos, y_oos,
                  alphas=None, gamma=1.0, n_components=500):

    if alphas is None:
        alphas = np.logspace(-4, 1, 20)

    best_alpha = None
    best_r2 = -np.inf

    for a in alphas:

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=42)),
            ("model", Lasso(alpha=a, max_iter=20000))
        ])

        pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

        y_hat_test = pipe.predict(X_test.to_numpy())
        r2 = get_OOS(y_test, y_train, y_hat_test)

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = a

    X_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_full = np.hstack([np.asarray(y_train).reshape(-1),
                        np.asarray(y_test).reshape(-1)])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=42)),
        ("model", Lasso(alpha=best_alpha, max_iter=20000))
    ])

    final_pipe.fit(X_full, y_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r2_oos = get_OOS(y_oos, y_full, y_hat_oos)

    return ["rbf_lasso", float(best_r2), float(r2_oos)], y_hat_oos


def rbf_elasticnet_fit(X_train, y_train, X_test, y_test, X_oos, y_oos,
                       alphas=None, l1_ratios=None,
                       gamma=1.0, n_components=500):

    if alphas is None:
        alphas = np.logspace(-4, 1, 15)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    best_alpha = None
    best_l1 = None
    best_r2 = -np.inf

    for a in alphas:
        for l1 in l1_ratios:

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=42)),
                ("model", ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000))
            ])

            pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

            y_hat_test = pipe.predict(X_test.to_numpy())
            r2 = get_OOS(y_test, y_train, y_hat_test)

            if r2 > best_r2:
                best_r2 = r2
                best_alpha = a
                best_l1 = l1

    X_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_full = np.hstack([np.asarray(y_train).reshape(-1),
                        np.asarray(y_test).reshape(-1)])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=42)),
        ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=20000))
    ])

    final_pipe.fit(X_full, y_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r2_oos = get_OOS(y_oos, y_full, y_hat_oos)

    return ["rbf_elastic_net", float(best_r2), float(r2_oos)], y_hat_oos


def pls_rbf_fit(X_train, y_train, X_test, y_test, X_oos, y_oos,
                n_components_list=None, gamma=1.0, rbf_components=500):

    if n_components_list is None:
        n_components_list = [5, 10, 20]

    best_k = None
    best_r2 = -np.inf

    for k in n_components_list:

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rbf", RBFSampler(gamma=gamma, n_components=rbf_components, random_state=42)),
            ("model", PLSRegression(n_components=k))
        ])

        pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1, 1))

        y_hat_test = pipe.predict(X_test.to_numpy()).reshape(-1)
        r2 = get_OOS(y_test, y_train, y_hat_test)

        if r2 > best_r2:
            best_r2 = r2
            best_k = k

    X_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_full = np.hstack([np.asarray(y_train).reshape(-1),
                        np.asarray(y_test).reshape(-1)])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(gamma=gamma, n_components=rbf_components, random_state=42)),
        ("model", PLSRegression(n_components=best_k))
    ])

    final_pipe.fit(X_full, y_full.reshape(-1, 1))

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r2_oos = get_OOS(y_oos, y_full, y_hat_oos)

    return ["pls_rbf", float(best_r2), float(r2_oos)], y_hat_oos


def gradient_boosting_fit(X_train, y_train, X_test, y_test, X_oos, y_oos,
                          learning_rates=None, n_estimators_list=None):

    if learning_rates is None:
        learning_rates = [0.01, 0.05, 0.1]
    if n_estimators_list is None:
        n_estimators_list = [100, 200]

    best_params = None
    best_r2 = -np.inf

    for lr in learning_rates:
        for n in n_estimators_list:

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(
                    learning_rate=lr,
                    n_estimators=n,
                    random_state=42
                ))
            ])

            pipe.fit(X_train.to_numpy(), np.asarray(y_train).reshape(-1))

            y_hat_test = pipe.predict(X_test.to_numpy())
            r2 = get_OOS(y_test, y_train, y_hat_test)

            if r2 > best_r2:
                best_r2 = r2
                best_params = (lr, n)

    X_full = np.vstack([X_train.to_numpy(), X_test.to_numpy()])
    y_full = np.hstack([np.asarray(y_train).reshape(-1),
                        np.asarray(y_test).reshape(-1)])

    final_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            learning_rate=best_params[0],
            n_estimators=best_params[1],
            random_state=42
        ))
    ])

    final_pipe.fit(X_full, y_full)

    y_hat_oos = final_pipe.predict(X_oos.to_numpy()).reshape(-1)
    r2_oos = get_OOS(y_oos, y_full, y_hat_oos)

    return ["gradient_boosting", float(best_r2), float(r2_oos)], y_hat_oos

def run_ridge_lasso(df):

    cut = pd.to_datetime("2004-01-01")

    X_train = df.loc[df.index < cut].copy()
    X_oos   = df.loc[df.index >= cut].copy()

    y_train = np.ones(len(X_train))

    # drop all-NaN columns
    cols_all_nan = X_train.columns[X_train.isna().all()]
    X_train = X_train.drop(columns=cols_all_nan)
    X_oos   = X_oos.drop(columns=cols_all_nan)

    results = {}

    for model_name, model, grid in [
        ("ridge", Ridge(fit_intercept=False), {"model__alpha": np.logspace(-4, 6, 40)}),
        ("lasso", Lasso(fit_intercept=False, max_iter=300000), {"model__alpha": np.logspace(-6, 1, 40)})
    ]:

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", model)
        ])

        cv = GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
        cv.fit(X_train, y_train)

        best = cv.best_estimator_
        w = best.named_steps["model"].coef_
        w = w / (np.sum(np.abs(w)) + 1e-12)

        X_oos_t = best.named_steps["imputer"].transform(X_oos)
        X_oos_t = best.named_steps["scaler"].transform(X_oos_t)

        r_p = X_oos_t @ w
        std = np.std(r_p, ddof=1)
        sharpe = np.sqrt(12) * np.mean(r_p) / std if std > 0 else np.nan

        results[model_name] = sharpe

    return results

def compute_monthly_coverage(df, min_threshold=None):

    char_cols = [c for c in df.columns 
                 if c not in ["permno", "yyyymm", "ret"]]

    df_indicator = df.copy()
    df_indicator[char_cols] = df_indicator[char_cols].notna().astype(int)

    monthly_counts = (
        df_indicator
            .groupby("yyyymm")[char_cols]
            .sum()
            .sort_index()
    )

    monthly_mean = pd.DataFrame(
        monthly_counts.mean(axis=1),
        columns=["avg_assets"]
    )

    valid_metrics = None
    if min_threshold is not None:
        valid_metrics = monthly_counts.columns[
            monthly_counts.min() >= min_threshold
        ].tolist()
        return monthly_counts, monthly_mean, valid_metrics

    return monthly_counts, monthly_mean

def build_decile_portfolio(df, characteristic):

    df = df[["permno", "yyyymm", "ret", characteristic]].dropna()

    df["decile"] = (
        df.groupby("yyyymm")[characteristic]
          .transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1)
    )

    decile_returns = (
        df.groupby(["yyyymm", "decile"])["ret"]
          .mean()
          .unstack()
          .sort_index()
    )

    long_short = decile_returns[10] - decile_returns[1]

    return decile_returns, long_short

def _safe_sharpe(r, ann=12, eps=1e-12):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 3:
        return np.nan
    sd = r.std(ddof=1)
    if sd < eps:
        return 0.0
    return r.mean() / sd * np.sqrt(ann)

# def run_ridge_lasso_grid_no_scaler(ls_df, split_date="2004-01-01", ann=12, n_splits=5):

#     X = ls_df.copy()
#     X.index = pd.to_datetime(X.index)

#     train = X[X.index < split_date]
#     test  = X[X.index >= split_date]

#     X_train = train.values
#     X_test  = test.values
#     y_train = np.ones(len(train))

#     cv = TimeSeriesSplit(n_splits=n_splits)
#     scoring = "neg_mean_squared_error"

#     ridge_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
#         ("model", Ridge(fit_intercept=False))
#     ])

#     lasso_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
#         ("model", Lasso(fit_intercept=False, max_iter=500000, tol=1e-6))
#     ])

#     ridge_grid = {"model__alpha": np.logspace(-10, 2, 80)}
#    # lasso_grid = {"model__alpha": np.logspace(-10, 0, 80)}
#     lasso_grid = {"model__alpha": np.logspace(-8, -3, 60)}

#     ridge_cv = GridSearchCV(ridge_pipe, ridge_grid, cv=cv, scoring=scoring, n_jobs=-1)
#     ridge_cv.fit(X_train, y_train)

#     lasso_cv = GridSearchCV(lasso_pipe, lasso_grid, cv=cv, scoring=scoring, n_jobs=-1)
#     lasso_cv.fit(X_train, y_train)

#     ridge_best = ridge_cv.best_estimator_
#     lasso_best = lasso_cv.best_estimator_

#     ridge_ret = ridge_best.predict(X_test)
#     lasso_ret = lasso_best.predict(X_test)

#     ridge_sh = _safe_sharpe(ridge_ret, ann=ann)
#     lasso_sh = _safe_sharpe(lasso_ret, ann=ann)

#     lasso_coef = lasso_best.named_steps["model"].coef_
#     lasso_nonzero = int((np.abs(lasso_coef) > 1e-12).sum())

#     return {
#         "ridge_alpha": ridge_cv.best_params_["model__alpha"],
#         "lasso_alpha": lasso_cv.best_params_["model__alpha"],
#         "ridge_sharpe": ridge_sh,
#         "lasso_sharpe": lasso_sh,
#         "lasso_nonzero": lasso_nonzero,
#     }

def run_ridge_lasso_grid_no_scaler(ls_df, split_date="2004-01-01", ann=12, n_splits=5):

    X = ls_df.copy()
    X.index = pd.to_datetime(X.index)

    train = X[X.index < split_date]
    test  = X[X.index >= split_date]

    X_train = train.values
    X_test  = test.values
    y_train = np.ones(len(train))

    cv = TimeSeriesSplit(n_splits=n_splits)
    scoring = "neg_mean_squared_error"

    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("model", Ridge(fit_intercept=False))
    ])

    lasso_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("model", Lasso(fit_intercept=False, max_iter=500000, tol=1e-6))
    ])

    ridge_grid = {"model__alpha": np.logspace(-10, 2, 80)}
    lasso_grid = {"model__alpha": np.logspace(-8, -3, 60)}

    ridge_cv = GridSearchCV(ridge_pipe, ridge_grid, cv=cv, scoring=scoring, n_jobs=-1)
    ridge_cv.fit(X_train, y_train)

    lasso_cv = GridSearchCV(lasso_pipe, lasso_grid, cv=cv, scoring=scoring, n_jobs=-1)
    lasso_cv.fit(X_train, y_train)

    ridge_best = ridge_cv.best_estimator_
    lasso_best = lasso_cv.best_estimator_

    ridge_pred = ridge_best.predict(X_test)
    lasso_pred = lasso_best.predict(X_test)

    ridge_ret = pd.Series(ridge_pred, index=test.index, name="ridge_ls")
    lasso_ret = pd.Series(lasso_pred, index=test.index, name="lasso_ls")

    ridge_sh = _safe_sharpe(ridge_ret.values, ann=ann)
    lasso_sh = _safe_sharpe(lasso_ret.values, ann=ann)

    lasso_coef = lasso_best.named_steps["model"].coef_
    lasso_nonzero = int((np.abs(lasso_coef) > 1e-12).sum())

    return {
        "ridge_alpha": ridge_cv.best_params_["model__alpha"],
        "lasso_alpha": lasso_cv.best_params_["model__alpha"],
        "ridge_sharpe": ridge_sh,
        "lasso_sharpe": lasso_sh,
        "lasso_nonzero": lasso_nonzero,
        "ridge_ls_ret": ridge_ret,
        "lasso_ls_ret": lasso_ret,
        "ridge_coef": ridge_best.named_steps["model"].coef_,
        "lasso_coef": lasso_coef,
    }