"""Müşteri DNA – RFM + K-Means müşteri segmentasyonu.

İki katman:
  1. RFM skorları (1-5) ile kural tabanlı, iş dilinde segmentler
     (En Değerli Müşteriler, Sadık Müşteriler, Gelişen Müşteriler, Yeni Gözlenen Müşteriler, Riskli Müşteriler, Pasif Müşteriler...).
  2. log-ölçekli R/F/M + kategori çeşitliliği üzerinde K-Means (k silhouette ile seçilir).
Ayrıca Müşteri DNA × Portföy DNA çapraz tablosu üretir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEGMENT_ORDER = ["En Değerli Müşteriler", "Sadık Müşteriler", "Gelişen Müşteriler", "Yeni Gözlenen Müşteriler",
                 "İlgi Gerektiren Müşteriler", "Riskli Müşteriler", "Riskli Değerli Müşteriler", "Pasif Müşteriler"]
SEGMENT_ACTIONS = {
    "En Değerli Müşteriler": "En değerli ve aktif: yeni çıkanları ilk bunlara sun, özel fiyat/öncelikli tedarik.",
    "Sadık Müşteriler": "Düzenli alım: çapraz satış (Sepet Analizi) ve hacim teşvikleri.",
    "Gelişen Müşteriler": "Yakın zamanda ve orta sıklıkta: sadakat programına dahil et.",
    "Yeni Gözlenen Müşteriler": "Veride ilk kez 2020'nin son 90 gününde görüldü: karşılama teklifi ve katalog tanıtımı.",
    "İlgi Gerektiren Müşteriler": "Ortalama değerler, düşüş sinyali: kişiselleştirilmiş teklif.",
    "Riskli Müşteriler": "Eskiden sık alıyordu, uzun süredir yok: geri kazanım araması.",
    "Riskli Değerli Müşteriler": "Yüksek ciro ama uzun süredir alım yok: yönetici seviyesinde ziyaret.",
    "Pasif Müşteriler": "Düşük değer ve uzun süredir pasif: düşük maliyetli kampanya.",
}


def rfm_table(df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
    ref = pd.Timestamp(ref_date) if ref_date is not None else df["InvoiceDate"].max() + pd.Timedelta(days=1)
    inv_col = "InvoiceID" if "InvoiceID" in df.columns else "InvoiceNo"
    g = df.groupby("CustomerID")
    r = pd.DataFrame({
        "Recency": (ref - g["InvoiceDate"].max()).dt.days,
        "Frequency": g[inv_col].nunique(),
        "Monetary": g["Revenue"].sum(),
        "Quantity": g["Quantity"].sum(),
        "Tenure_Days": (ref - g["InvoiceDate"].min()).dt.days,
        "Categories": g["Category"].nunique(),
        "Products": g["StockCode"].nunique(),
        "Avg_Discount": g["Discount"].mean(),
    })
    first = df.drop_duplicates("CustomerID").set_index("CustomerID")
    r["City"] = first["City"]
    r["CustomerType"] = first["CustomerType"]
    r["Avg_Order_Value"] = r["Monetary"] / r["Frequency"]

    r["R"] = pd.qcut(r["Recency"].rank(method="first", ascending=False), 5, labels=False) + 1
    r["F"] = pd.qcut(r["Frequency"].rank(method="first"), 5, labels=False) + 1
    r["M"] = pd.qcut(r["Monetary"].rank(method="first"), 5, labels=False) + 1
    r["RFM_Score"] = r["R"].astype(str) + r["F"].astype(str) + r["M"].astype(str)
    r["Segment"] = _segment(r)
    r["Action"] = r["Segment"].map(SEGMENT_ACTIONS)
    return r


def _segment(r: pd.DataFrame) -> pd.Series:
    R, F, M = r["R"], r["F"], r["M"]
    fm = (F + M) / 2
    conds = [
        (r["Tenure_Days"] <= 90),
        (R >= 4) & (fm >= 4),
        (R >= 3) & (fm >= 3.5),
        (R >= 3) & (fm >= 2),
        (R <= 2) & (fm >= 4.5),
        (R <= 2) & (fm >= 3),
        (R == 3),
        (R >= 4),
    ]
    choices = ["Yeni Gözlenen Müşteriler", "En Değerli Müşteriler", "Sadık Müşteriler", "Gelişen Müşteriler",
               "Riskli Değerli Müşteriler", "Riskli Müşteriler", "İlgi Gerektiren Müşteriler", "Gelişen Müşteriler"]
    return pd.Series(np.select(conds, choices, default="Pasif Müşteriler"), index=r.index)


def kmeans_segments(r: pd.DataFrame, k_range=range(3, 9), seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """K-Means kümeleri; k silhouette ile seçilir. Kümeler merkezlerine göre adlandırılır."""
    feats = pd.DataFrame({
        "log_R": np.log1p(r["Recency"]),
        "log_F": np.log1p(r["Frequency"]),
        "log_M": np.log1p(r["Monetary"].clip(lower=0)),
        "log_Cat": np.log1p(r["Categories"]),
    }, index=r.index)
    X = StandardScaler().fit_transform(feats)
    scores = {}
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, lab, random_state=seed))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    out = r.copy()
    out["Cluster"] = km.labels_

    prof = out.groupby("Cluster").agg(Customers=("R", "size"), Recency=("Recency", "median"),
                                      Frequency=("Frequency", "median"), Monetary=("Monetary", "median"),
                                      Categories=("Categories", "median"))
    order = (prof["Frequency"].rank() + prof["Monetary"].rank()).rank(method="first") - 1
    val_rank = (order + 0.5) / len(prof)
    names = {}
    for c, row in prof.iterrows():
        value = "Yüksek Değer" if val_rank[c] > 0.66 else ("Orta Değer" if val_rank[c] > 0.33 else "Düşük Değer")
        active = "Aktif" if row["Recency"] <= 30 else ("Soğuyan" if row["Recency"] <= 90 else "Pasif")
        names[c] = f"{value} · {active}"
    seen = {}
    for c in names:
        seen[names[c]] = seen.get(names[c], 0) + 1
        if list(names.values()).count(names[c]) > 1:
            names[c] = f"{names[c]} ({seen[names[c]]})"
    out["Cluster_Name"] = out["Cluster"].map(names)
    prof["Cluster_Name"] = prof.index.map(names)
    return out, {"silhouette": scores, "best_k": best_k, "profile": prof.round(1)}


def segment_summary(r: pd.DataFrame, col: str = "Segment") -> pd.DataFrame:
    s = r.groupby(col).agg(Customers=("Monetary", "size"), Revenue=("Monetary", "sum"),
                           Recency_med=("Recency", "median"), Frequency_med=("Frequency", "median"),
                           AOV_med=("Avg_Order_Value", "median"))
    s["Customer_Share_%"] = 100 * s["Customers"] / s["Customers"].sum()
    s["Revenue_Share_%"] = 100 * s["Revenue"] / s["Revenue"].sum()
    if col == "Segment":
        s = s.reindex([x for x in SEGMENT_ORDER if x in s.index])
        s["Action"] = s.index.map(SEGMENT_ACTIONS)
    return s.round(1)


def segment_x_portfolio(df: pd.DataFrame, rfm: pd.DataFrame, portfolio: pd.DataFrame,
                        seg_col: str = "Segment") -> pd.DataFrame:
    """Her müşteri segmentinin cirosunun Portföy DNA gruplarına dağılımı (% satır)."""
    m = df[["CustomerID", "StockCode", "Revenue"]].merge(rfm[[seg_col]], left_on="CustomerID", right_index=True)
    m = m.merge(portfolio[["DNA"]], left_on="StockCode", right_index=True)
    t = m.pivot_table(index=seg_col, columns="DNA", values="Revenue", aggfunc="sum", fill_value=0)
    t = 100 * t.div(t.sum(axis=1), axis=0)
    from .portfolio import DNA_ORDER
    order = [x for x in SEGMENT_ORDER if x in t.index] if seg_col == "Segment" else t.index
    return t.reindex(index=order, columns=[c for c in DNA_ORDER if c in t.columns]).round(1)


def missing_orders(df: pd.DataFrame, ref_date=None, min_orders: int = 5, factor: float = 2.0,
                   min_days: int = 21) -> pd.DataFrame:
    """Kaçan Sipariş Dedektörü: normal sipariş aralığının çok üzerinde süredir sipariş vermeyen müşteriler.

    Her müşterinin sipariş tarihleri arasındaki medyan gün farkı hesaplanır. Son siparişten bu yana geçen süre
    bu medyanın `factor` katını aşıyorsa müşteri işaretlenir (yalnızca en az `min_orders` sipariş günü olanlar).
    Çok sık sipariş verenlerde gürültüyü önlemek için aralık en az 7 gün sayılır ve en az `min_days` gün
    sessizlik aranır.
    Bu bir risk sinyalidir; müşterinin kaybedildiği anlamına gelmez.
    """
    ref = pd.Timestamp(ref_date) if ref_date is not None else df["InvoiceDate"].max()
    days = df[["CustomerID", "InvoiceDate"]].drop_duplicates().sort_values(["CustomerID", "InvoiceDate"])
    days["gap"] = days.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    g = days.groupby("CustomerID").agg(Order_Days=("InvoiceDate", "size"), Typical_Gap=("gap", "median"),
                                      Last_Purchase=("InvoiceDate", "max"))
    rev = df.groupby("CustomerID")["Revenue"].sum()
    g["Revenue"] = rev
    g["Days_Since_Last"] = (ref - g["Last_Purchase"]).dt.days
    g = g[(g["Order_Days"] >= min_orders) & g["Typical_Gap"].notna()]
    g["Delay_Ratio"] = g["Days_Since_Last"] / g["Typical_Gap"].clip(lower=7)
    g["Expected_By"] = g["Last_Purchase"] + pd.to_timedelta(g["Typical_Gap"], unit="D")
    first = df.drop_duplicates("CustomerID").set_index("CustomerID")
    g["City"] = first["City"]
    out = g[(g["Delay_Ratio"] >= factor) & (g["Days_Since_Last"] >= min_days)].sort_values("Revenue", ascending=False)
    return out.round({"Typical_Gap": 1, "Delay_Ratio": 1, "Revenue": 2})
