"""Talep Tahmini – haftalık talep tahmini (kategori veya ürün bazında).

ÖNEMLİ BAĞLAM: Veri yalnızca 2020'yi (≈51 tam hafta) kapsıyor ve Covid kapanma
dönemini içeriyor. Yıllık mevsimsellik öğrenilemez ve rejim değişiklikleri sert.
Bu yüzden modül "en karmaşık modeli" değil, **doğrulamada en iyi çıkan modeli** seçer:

  Adaylar
    * Naive        : geçen haftanın değeri
    * MA_4         : son 4 haftanın ortalaması
    * SES          : basit üstel düzleştirme (alpha her seri için train'de seçilir)
    * GBM          : ölçeklenmiş gecikmelerle global gradient boosting
                     (xgboost varsa XGBRegressor, yoksa HistGradientBoostingRegressor)
  Değerlendirme
    * Son `test_weeks` hafta ayrı tutulur, 1 hafta ileri kayan tahmin
    * Bu pencerenin ilk yarısı yöntem SEÇİMİ, ikinci yarısı performans RAPORU için kullanılır
      (aynı dönemde hem seçip hem raporlamak sonucu iyimser gösterirdi)
    * MAE, RMSE, MAPE, WAPE
  Tahmin
    * Her seri için doğrulamada en düşük WAPE'yi veren yöntem kullanılır ("auto")
    * %80 tahmin aralığı: doğrulama hatalarının standart sapmasından
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 4, 8]
NORM_FEATS = [f"n{l}" for l in LAGS]
METHODS = ["Naive", "MA_4", "SES", "GBM"]


# ---------------------------------------------------------------------------
def weekly_series(df: pd.DataFrame, level: str = "Category", top_n: int = 20, items=None,
                  include_total: bool = True) -> pd.DataFrame:
    """Haftalık adet matrisi (satır: hafta başı Pazartesi, sütun: seri)."""
    week = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    if items is None:
        items = df.groupby(level)["Quantity"].sum().nlargest(top_n).index
    sub = df[df[level].isin(items)]
    w = sub.groupby([week[sub.index], level])["Quantity"].sum().unstack(fill_value=0)
    full = pd.date_range(week.min(), week.max(), freq="W-MON")
    w = w.reindex(full, fill_value=0)
    if include_total and level == "Category":
        w["TOPLAM"] = df.groupby(week)["Quantity"].sum().reindex(full, fill_value=0)
    w.index.name = "Week"
    return w.iloc[1:-1]            


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    nz = y > 0
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAPE_%": float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100) if nz.any() else np.nan,
        "WAPE_%": float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1) * 100),
    }


# --- SES -------------------------------------------------------------------
def _ses_path(y: np.ndarray, alpha: float) -> np.ndarray:
    """1 adım ileri SES tahminleri (pred[t] = t'den önceki bilgiyle)."""
    pred = np.empty_like(y, dtype=float)
    level = y[0]
    for t in range(len(y)):
        pred[t] = level
        level = alpha * y[t] + (1 - alpha) * level
    return pred


def _best_alpha(y: np.ndarray) -> float:
    grid = np.linspace(0.1, 1.0, 10)
    errs = [np.abs(y[4:] - _ses_path(y, a)[4:]).sum() for a in grid]
    return float(grid[int(np.argmin(errs))])


# --- GBM -------------------------------------------------------------------
def _regressor(seed=42):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9,
                            random_state=seed, objective="reg:squarederror")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                             min_samples_leaf=20, random_state=seed)


def _gbm_frame(series: pd.DataFrame) -> pd.DataFrame:
    long = series.stack().rename("y").reset_index()
    long.columns = ["Week", "Series", "y"]
    long = long.sort_values(["Series", "Week"])
    g = long.groupby("Series")["y"]
    for l in LAGS:
        long[f"l{l}"] = g.shift(l)
    long["scale"] = g.transform(lambda s: s.shift(1).rolling(8, min_periods=4).mean()) + 1
    for l in LAGS:
        long[f"n{l}"] = long[f"l{l}"] / long["scale"]
    long["target"] = long["y"] / long["scale"]
    return long.dropna()


# ---------------------------------------------------------------------------
def evaluate(series: pd.DataFrame, test_weeks: int = 8) -> dict:
    """Son `test_weeks` hafta üzerinde tüm adayları 1 hafta ileri karşılaştırır."""
    cut = series.index[-test_weeks]
    test_idx = series.index >= cut
    preds = {m: pd.DataFrame(index=series.index[test_idx], columns=series.columns, dtype=float) for m in METHODS}

    for s in series.columns:
        y = series[s].to_numpy(float)
        preds["Naive"][s] = np.r_[np.nan, y[:-1]][test_idx]
        preds["MA_4"][s] = pd.Series(y).shift(1).rolling(4).mean().to_numpy()[test_idx]
        alpha = _best_alpha(y[~test_idx])
        preds["SES"][s] = _ses_path(y, alpha)[test_idx]

    long = _gbm_frame(series)
    tr, te = long[long["Week"] < cut], long[long["Week"] >= cut].copy()
    model = _regressor().fit(tr[NORM_FEATS], tr["target"])
    te["p"] = np.clip(model.predict(te[NORM_FEATS]), 0, None) * te["scale"]
    preds["GBM"] = te.pivot(index="Week", columns="Series", values="p").reindex(
        index=preds["Naive"].index, columns=series.columns)

    actual = series[test_idx]
    half = test_weeks // 2
    sel_idx, rep_idx = actual.index[:half], actual.index[half:]
    overall = pd.DataFrame({m: _metrics(actual.loc[rep_idx].to_numpy().ravel(), preds[m].loc[rep_idx].to_numpy().ravel())
                            for m in METHODS}).T.round(2)
    per = pd.DataFrame({m: [_metrics(actual.loc[sel_idx, s], preds[m].loc[sel_idx, s])["WAPE_%"] for s in series.columns]
                        for m in METHODS}, index=series.columns).round(1)
    per["Best"] = per[METHODS].idxmin(axis=1)
    per["Test_WAPE_%"] = [round(_metrics(actual.loc[rep_idx, s], preds[per.loc[s, "Best"]].loc[rep_idx, s])["WAPE_%"], 1)
                          for s in series.columns]
    sel_overall = {m: _metrics(actual.loc[sel_idx].to_numpy().ravel(), preds[m].loc[sel_idx].to_numpy().ravel())["WAPE_%"]
                   for m in METHODS}
    global_best = min(sel_overall, key=sel_overall.get)       # tüm seriler için tek yöntem (daha az aşırı uyum)
    chosen_pred = np.concatenate([preds[per.loc[s, "Best"]].loc[rep_idx, s].to_numpy() for s in series.columns])
    chosen_act = np.concatenate([actual.loc[rep_idx, s].to_numpy() for s in series.columns])
    overall.loc["Seri bazında seçim"] = pd.Series(_metrics(chosen_act, chosen_pred)).round(2)
    resid_sd = {s: float(np.nanstd(actual[s] - preds[per.loc[s, "Best"]][s])) for s in series.columns}
    return {"overall": overall, "per_series": per, "preds": preds, "actual": actual,
            "cutoff": str(cut.date()), "selection_window": (str(sel_idx[0].date()), str(sel_idx[-1].date())),
            "report_window": (str(rep_idx[0].date()), str(rep_idx[-1].date())),
            "gbm_name": type(model).__name__, "resid_sd": resid_sd, "global_best": global_best,
            "selection_wape": {k: round(v, 2) for k, v in sel_overall.items()}}


def forecast(series: pd.DataFrame, horizon: int = 8, evaluation: dict | None = None,
             method: str = "global") -> pd.DataFrame:
    """method: 'global' (seçim penceresinde genel olarak en iyi tek yöntem, varsayılan),
    'auto' (her seri için ayrı seçim) veya yöntem adı."""
    """Her seri için `horizon` hafta tahmin (uzun format: Week, Series, Forecast, Lower, Upper, Method)."""
    ev = evaluation or evaluate(series)
    future = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")

    gbm_model = None
    rows = []
    for s in series.columns:
        m = (ev["per_series"].loc[s, "Best"] if method == "auto"
             else ev["global_best"] if method == "global" else method)
        y = series[s].to_numpy(float)
        if m == "Naive":
            path = np.repeat(y[-1], horizon)
        elif m == "MA_4":
            path = np.repeat(y[-4:].mean(), horizon)
        elif m == "SES":
            a = _best_alpha(y)
            level = _ses_path(y, a)[-1] * (1 - a) + a * y[-1]
            path = np.repeat(level, horizon)
        else:  
            if gbm_model is None:
                long = _gbm_frame(series)
                gbm_model = _regressor().fit(long[NORM_FEATS], long["target"])
            hist = list(y)
            path = []
            for _ in range(horizon):
                h = np.array(hist)
                scale = h[-8:].mean() + 1
                x = pd.DataFrame([{f"n{l}": h[-l] / scale for l in LAGS}])
                v = float(np.clip(gbm_model.predict(x)[0], 0, None) * scale)
                path.append(v)
                hist.append(v)
            path = np.array(path)
        sd = float(np.nanstd(ev["actual"][s] - ev["preds"][m][s]))
        widen = np.sqrt(np.arange(1, horizon + 1))          
        for i, w in enumerate(future):
            rows.append({"Week": w, "Series": s, "Forecast": path[i],
                         "Lower": max(0.0, path[i] - 1.28 * sd * widen[i]),
                         "Upper": path[i] + 1.28 * sd * widen[i], "Method": m})
    return pd.DataFrame(rows).round({"Forecast": 0, "Lower": 0, "Upper": 0})


def outlook(series: pd.DataFrame, fc: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Önümüzdeki `weeks` hafta tahmini vs son `weeks` hafta gerçekleşen."""
    last = series.tail(weeks).sum()
    nxt = fc[fc["Week"].isin(sorted(fc["Week"].unique())[:weeks])].groupby("Series")["Forecast"].sum()
    method = fc.groupby("Series")["Method"].first()
    t = pd.DataFrame({"Son_Dönem": last, "Tahmin": nxt, "Yöntem": method})
    t["Değişim_%"] = (100 * (t["Tahmin"] - t["Son_Dönem"]) / t["Son_Dönem"].replace(0, np.nan)).round(1)
    return t.sort_values("Tahmin", ascending=False)
