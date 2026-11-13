"""Sepet Analizi – birlikte satın alma (A -> B) kuralları.

Apriori'nin 2'li kural adımı seyrek matris çarpımıyla hesaplanır (43 bin ürün için
mlxtend'in yoğun matrisine göre çok daha hızlı ve hafıza dostu). Metrikler:
    support(A,B)    = A ve B'yi birlikte içeren sepet oranı
    confidence(A→B) = P(B | A)
    lift(A→B)       = confidence / support(B)   (>1 => tesadüften fazla birliktelik)
Sepet = fatura. B2B'de çok büyük faturalar (yüzlerce kalem) sahte birliktelik ürettiği
için `max_basket_size` üzerindeki faturalar dışarıda bırakılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .data import product_lookup


def _basket_matrix(df: pd.DataFrame, item_col: str, max_basket_size: int | None):
    inv_col = "InvoiceID" if "InvoiceID" in df.columns else "InvoiceNo"
    pairs = df[[inv_col, item_col]].drop_duplicates()
    if max_basket_size:
        size = pairs.groupby(inv_col)[item_col].transform("size")
        pairs = pairs[(size >= 2) & (size <= max_basket_size)]
    inv_codes, inv_uni = pd.factorize(pairs[inv_col])
    item_codes, item_uni = pd.factorize(pairs[item_col])
    X = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (inv_codes, item_codes)),
                          shape=(len(inv_uni), len(item_uni)))
    return X, np.asarray(item_uni)


def association_rules(df: pd.DataFrame, level: str = "product", min_support_count: int = 20,
                      min_confidence: float = 0.05, min_lift: float = 1.2,
                      max_basket_size: int | None = 150) -> pd.DataFrame:
    """A→B kuralları. level: 'product' veya 'category'."""
    item_col = "StockCode" if level == "product" else "Category"
    X, items = _basket_matrix(df, item_col, max_basket_size)
    n_baskets = X.shape[0]
    item_count = np.asarray(X.sum(axis=0)).ravel()

    keep = np.where(item_count >= min_support_count)[0]       
    Xk = X[:, keep].tocsc()
    co = (Xk.T @ Xk).tocoo()
    mask = (co.row != co.col) & (co.data >= min_support_count)
    a, b, cnt = keep[co.row[mask]], keep[co.col[mask]], co.data[mask]

    rules = pd.DataFrame({
        "A": items[a], "B": items[b], "Count_AB": cnt.astype(int),
        "Support": cnt / n_baskets,
        "Confidence": cnt / item_count[a],
        "Support_B": item_count[b] / n_baskets,
    })
    rules["Lift"] = rules["Confidence"] / rules["Support_B"]
    rules = rules[(rules["Confidence"] >= min_confidence) & (rules["Lift"] >= min_lift)]

    if level == "product":
        lk = product_lookup(df)
        rules["A_Name"] = rules["A"].map(lk["Description"])
        rules["B_Name"] = rules["B"].map(lk["Description"])
        rules["A_Category"] = rules["A"].map(lk["Category"])
        rules["B_Category"] = rules["B"].map(lk["Category"])
    rules.attrs.update(n_baskets=n_baskets, level=level)
    return rules.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)


def cross_sell(rules: pd.DataFrame, item, n: int = 10) -> pd.DataFrame:
    """Bir ürün/kategori alan müşteriye önerilecek tamamlayıcılar."""
    r = rules[rules["A"] == item].copy()
    r["Score"] = r["Confidence"] * np.log1p(r["Lift"])
    return r.sort_values("Score", ascending=False).head(n)


def bundles(rules: pd.DataFrame, n: int = 20, min_confidence: float = 0.2,
            cross_category_only: bool = False) -> pd.DataFrame:
    """Paket önerisi: iki yönlü güçlü birliktelikler (A<->B tek satır).
    cross_category_only=True -> aynı serinin ciltleri yerine farklı kategorilerden eşleşmeler."""
    r = rules[rules["Confidence"] >= min_confidence].copy()
    if cross_category_only and "A_Category" in r.columns:
        r = r[r["A_Category"] != r["B_Category"]]
    r["key"] = [tuple(sorted((str(x), str(y)))) for x, y in zip(r["A"], r["B"])]
    r = r.sort_values("Lift", ascending=False).drop_duplicates("key").drop(columns="key")
    return r.head(n)
