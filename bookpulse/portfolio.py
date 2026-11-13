"""Portföy DNA + Portföy Optimizasyonu.

İlk versiyona göre değişiklikler:
  * CV, ürünün İLK SATIŞ AYINDAN itibaren hesaplanır (yaşam döngüsü). Yıl içinde
    çıkan ürünlerin satışa girmeden önceki ayları artık 0 sayılmaz.
    Karşılaştırma için takvim bazlı CV de (CV_Calendar) tutulur.
  * Eşikler config'den gelir (varsayılan medyan) ve sonuçla birlikte döndürülür.
  * Her ürüne performans puanı (0-100) ve aksiyon önerisi eklenir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import product_lookup

DNA_ORDER = ["Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep", "Az Satan – Düzenli Talep", "Az Satan – Değişken Talep", "Yetersiz Satış Geçmişi"]
from .ui import DNA_COLORS  

ACTIONS = {
    "Sürekli Bulunurluğu Değerlendir": "Sürekli ve yüksek talep: ürünün kesintisiz bulunurluğu öncelikli değerlendirilmeli.",
    "Sezon / Kampanya Planla": "Yüksek ama dalgalı talep: zirve aylarından önce stok ve kampanya planla.",
    "Hedefli Satış": "Düşük ama düzenli talep: bu ürünü alan sadık müşteri grubuna odaklan.",
    "Tanıtım Fırsatı": "Satışı yükselişte ama müşteri erişimi dar: görünürlüğü artır.",
    "İzle (Düşüşte)": "Güçlü ürün ama son dönemde düşüşte: nedenini araştır.",
    "Stok Politikasını Gözden Geçir": "Düşük ve öngörülemez talep: siparişe göre tedarik et.",
    "Yeni Ürün – İzle": "Son 3 ayda satışa girdi: gelişimini takip et.",
    "Ürün Performansını İncele": "Çok düşük satış ve 90+ gündür hareketsiz: ürünün portföydeki yerini incele.",
}


# ---------------------------------------------------------------------------
def monthly_matrix(df: pd.DataFrame, value: str = "Quantity") -> pd.DataFrame:
    m = df.pivot_table(index="StockCode", columns="Month", values=value, aggfunc="sum", fill_value=0)
    all_months = pd.period_range(df["Month"].min(), df["Month"].max(), freq="M")
    return m.reindex(columns=all_months, fill_value=0)


def _lifecycle_stats(mat: pd.DataFrame) -> pd.DataFrame:
    arr = mat.to_numpy(dtype=float)
    n_months = arr.shape[1]
    has = arr > 0
    first_idx = np.where(has.any(axis=1), has.argmax(axis=1), n_months - 1)
    mask = np.arange(n_months)[None, :] >= first_idx[:, None]      
    window = mask.sum(axis=1)
    vals = np.where(mask, arr, np.nan)
    mean = np.nanmean(vals, axis=1)
    std = np.nanstd(vals, axis=1)                                     
    last_idx = n_months - 1 - has[:, ::-1].argmax(axis=1)
    cal_mean, cal_std = arr.mean(axis=1), arr.std(axis=1)
    return pd.DataFrame({
        "Total_Qty": arr.sum(axis=1),
        "Window_Months": window,
        "Monthly_Avg": mean,
        "Monthly_Std": std,
        "CV": np.where(mean > 0, std / np.where(mean > 0, mean, 1), np.nan),
        "CV_Calendar": np.where(cal_mean > 0, cal_std / np.where(cal_mean > 0, cal_mean, 1), np.nan),
        "Active_Months": has.sum(axis=1),
        "First_Month": mat.columns[first_idx].astype(str),
        "Last_Month": mat.columns[last_idx].astype(str),
        "Peak_Month": mat.columns[arr.argmax(axis=1)].astype(str),
        "Qty_Last3": arr[:, -3:].sum(axis=1),
        "Qty_Prev3": arr[:, -6:-3].sum(axis=1),
    }, index=mat.index)


def build_portfolio(df: pd.DataFrame, volume_threshold=None, cv_threshold=None) -> tuple[pd.DataFrame, dict]:
    """Ürün bazlı Portföy DNA tablosu ve kullanılan eşikleri döndürür."""
    volume_threshold = volume_threshold if volume_threshold is not None else config.VOLUME_THRESHOLD
    cv_threshold = cv_threshold if cv_threshold is not None else config.CV_THRESHOLD

    p = _lifecycle_stats(monthly_matrix(df))
    agg = df.groupby("StockCode").agg(
        Revenue=("Revenue", "sum"), Customers=("CustomerID", "nunique"),
        Orders=("InvoiceID", "nunique"), Avg_Discount=("Discount", "mean"),
        Last_Sale=("InvoiceDate", "max"),
    )
    p = p.join(agg).join(product_lookup(df))

    limited = (p["Active_Months"] < config.LIMITED_MIN_ACTIVE_MONTHS) | (p["Total_Qty"] < config.LIMITED_MIN_TOTAL_QTY)
    core = p[~limited]
    vt = float(core["Total_Qty"].median()) if volume_threshold is None else float(volume_threshold)
    ct = float(core["CV"].median()) if cv_threshold is None else float(cv_threshold)

    high_vol = p["Total_Qty"] >= vt
    low_cv = p["CV"] <= ct
    p["DNA"] = np.select(
        [limited, high_vol & low_cv, high_vol & ~low_cv, ~high_vol & low_cv],
        ["Yetersiz Satış Geçmişi", "Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep", "Az Satan – Düzenli Talep"],
        default="Az Satan – Değişken Talep",
    )

    p = _optimize(p, df["InvoiceDate"].max())
    thresholds = {"volume_threshold": vt, "cv_threshold": ct,
                  "limited_min_active_months": config.LIMITED_MIN_ACTIVE_MONTHS,
                  "limited_min_total_qty": config.LIMITED_MIN_TOTAL_QTY}
    return p, thresholds


# ---------------------------------------------------------------------------
def _pct(s: pd.Series, ascending=True) -> pd.Series:
    return s.rank(pct=True, ascending=ascending).fillna(0)


def _optimize(p: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    p = p.copy()
    p["Trend"] = ((p["Qty_Last3"] - p["Qty_Prev3"]) / (p["Qty_Prev3"] + 5)).clip(-1, 3)
    p["Days_Since_Sale"] = (ref_date - p["Last_Sale"]).dt.days

    p["Performance_Score"] = (100 * (
        0.35 * _pct(p["Revenue"])
        + 0.25 * _pct(p["Customers"])
        + 0.20 * _pct(p["Trend"])
        + 0.20 * _pct(p["CV"], ascending=False)
    )).round(1)

    first = pd.PeriodIndex(p["First_Month"], freq="M")
    last_period = pd.Period(ref_date, freq="M")
    is_new = (last_period - first).map(lambda x: x.n) < 3
    reach_low = p["Customers"] <= p["Customers"].median()

    conds = [
        (p["DNA"] == "Çok Satan – Düzenli Talep") & (p["Trend"] < -0.3),
        (p["DNA"] == "Çok Satan – Düzenli Talep"),
        (p["DNA"] == "Çok Satan – Değişken Talep"),
        (p["DNA"].isin(["Az Satan – Düzenli Talep", "Az Satan – Değişken Talep"])) & (p["Trend"] > 0.5) & reach_low,
        (p["DNA"] == "Az Satan – Düzenli Talep"),
        (p["DNA"] == "Az Satan – Değişken Talep"),
        (p["DNA"] == "Yetersiz Satış Geçmişi") & is_new,
        (p["DNA"] == "Yetersiz Satış Geçmişi") & (p["Days_Since_Sale"] > 90),
    ]
    choices = ["İzle (Düşüşte)", "Sürekli Bulunurluğu Değerlendir", "Sezon / Kampanya Planla", "Tanıtım Fırsatı",
               "Hedefli Satış", "Stok Politikasını Gözden Geçir", "Yeni Ürün – İzle", "Ürün Performansını İncele"]
    p["Action"] = np.select(conds, choices, default="Stok Politikasını Gözden Geçir")
    p["Action_Reason"] = p["Action"].map(ACTIONS)
    return p


# ---------------------------------------------------------------------------
def dna_summary(p: pd.DataFrame) -> pd.DataFrame:
    s = p.groupby("DNA").agg(Products=("DNA", "size"), Quantity=("Total_Qty", "sum"), Revenue=("Revenue", "sum"))
    s["Product_Share_%"] = 100 * s["Products"] / s["Products"].sum()
    s["Qty_Share_%"] = 100 * s["Quantity"] / s["Quantity"].sum()
    s["Revenue_Share_%"] = 100 * s["Revenue"] / s["Revenue"].sum()
    return s.reindex(DNA_ORDER).round(2)


def action_summary(p: pd.DataFrame) -> pd.DataFrame:
    s = p.groupby("Action").agg(Products=("Action", "size"), Revenue=("Revenue", "sum"))
    s["Revenue_Share_%"] = (100 * s["Revenue"] / s["Revenue"].sum()).round(2)
    s["Açıklama"] = s.index.map(ACTIONS)
    return s.sort_values("Revenue", ascending=False)


def category_concentration(p: pd.DataFrame, top: int | None = 30) -> pd.DataFrame:
    """Kategori bazında portföy yoğunlaşması (HHI ve ilk %10 ürünün ciro payı)."""
    def _hhi(r):
        sh = r / r.sum() if r.sum() > 0 else r
        return float((sh ** 2).sum() * 10000)

    def _top10(r):
        r = r.sort_values(ascending=False)
        k = max(1, int(np.ceil(len(r) * 0.1)))
        return float(r.iloc[:k].sum() / r.sum() * 100) if r.sum() > 0 else 0.0

    g = p.groupby("Category")
    out = pd.DataFrame({
        "Products": g.size(),
        "Revenue": g["Revenue"].sum(),
        "Stable_Stars": g["DNA"].apply(lambda s: (s == "Çok Satan – Düzenli Talep").sum()),
        "Limited_History_%": g["DNA"].apply(lambda s: 100 * (s == "Yetersiz Satış Geçmişi").mean()),
        "HHI": g["Revenue"].apply(_hhi),
        "Top10%_Product_Rev_Share": g["Revenue"].apply(_top10),
    })
    out["Revenue_Share_%"] = 100 * out["Revenue"] / out["Revenue"].sum()
    out = out.sort_values("Revenue", ascending=False).round(1)
    return out.head(top) if top else out
