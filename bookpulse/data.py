"""Veri yükleme ve temizlik.

İlk versiyondaki (homework1.ipynb) eksiklere göre yapılan düzeltmeler:
  1. Kategori / şehir adlarındaki baş-son boşluklar temizlenir ("Aile - Çocuk " -> "Aile - Çocuk").
  2. "A00001-01" gibi alt hesaplar ana müşteriye bağlanır (CustomerID); orijinal kod Branch'te kalır.
  3. Fatura öneki DocType olarak çıkarılır; satış dışı belge türleri (MUH) analizden çıkarılır.
  4. Sıfır / negatif NetPrice ve sembolik UnitPrice (<0.01) satırları çıkarılır.
  5. Excel'in sayı/tarihe çevirdiği ürün adları metne döndürülür ve işaretlenir.
  6. Birebir tekrar eden satırlar korunur ama IsDuplicate ile işaretlenir.
  7. Boş şehir, aynı müşterinin diğer kayıtlarından doldurulur; kalmazsa "Unknown".
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
def load_raw(path: str | Path = config.RAW_XLSX) -> pd.DataFrame:
    """Ham Excel dosyasını okur (40 MB, ~1 dk sürebilir)."""
    return pd.read_excel(path, sheet_name=0)


def _description_to_text(v) -> tuple[str, bool]:
    """Excel'in otomatik çevirdiği değerleri metne döndürür. (metin, şüpheli_mi)"""
    if isinstance(v, str):
        return v.strip(), False
    if isinstance(v, (_dt.datetime, pd.Timestamp)):
        return v.strftime("%d.%m.%Y"), True
    if isinstance(v, _dt.time):
        return v.strftime("%H:%M"), True
    if isinstance(v, float) and np.isnan(v):
        return "", True
    if isinstance(v, float) and v.is_integer():
        return str(int(v)), True
    return str(v), True


def clean(raw: pd.DataFrame, *, sales_doc_types=None, merge_branches=None) -> tuple[pd.DataFrame, dict]:
    """Ham veriyi temizler. (df_clean, rapor) döndürür."""
    sales_doc_types = sales_doc_types or config.SALES_DOC_TYPES
    merge_branches = config.MERGE_CUSTOMER_BRANCHES if merge_branches is None else merge_branches

    df = raw.copy()
    report: dict = {"Ham satır": len(df)}

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["InvoiceNo", "CustomerID", "Category", "City"]:
        df[col] = df[col].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("int64")

    desc = df["Description"].map(_description_to_text)
    df["Description"] = desc.str[0]
    df["DescriptionSuspect"] = desc.str[1].astype(bool)
    report["Metne döndürülen ürün adı"] = int(df["DescriptionSuspect"].sum())

    df["DocType"] = df["InvoiceNo"].str[:3]
    df["Branch"] = df["CustomerID"]
    if merge_branches:
        df["CustomerID"] = df["CustomerID"].str.split("-").str[0]
    df["CustomerType"] = df["CustomerID"].str[0]          
    report["Ana müşteriye bağlanan alt hesap satırı"] = int((df["Branch"] != df["CustomerID"]).sum())

    base_cols = ["InvoiceNo", "Branch", "InvoiceDate", "StockCode", "Quantity", "NetPrice", "UnitPrice"]
    df["IsDuplicate"] = df.duplicated(subset=base_cols, keep="first")
    report["İşaretlenen tekrar satır"] = int(df["IsDuplicate"].sum())

    reasons = pd.Series("", index=df.index, dtype="object")
    reasons[df["InvoiceDate"].isna()] = "tarih yok"
    reasons[df["Quantity"].isna()] = "miktar yok"
    reasons[(reasons == "") & (df["Quantity"] <= 0)] = "miktar <= 0"
    reasons[(reasons == "") & (df["UnitPrice"] < config.MIN_UNIT_PRICE)] = "sembolik birim fiyat"
    reasons[(reasons == "") & (df["NetPrice"] <= 0)] = "net fiyat <= 0"
    reasons[(reasons == "") & (~df["DocType"].isin(sales_doc_types))] = "satış dışı belge türü"
    report["Çıkarılan satırlar"] = reasons[reasons != ""].value_counts().to_dict()
    df = df[reasons == ""].copy()

    df["PriceAnomaly"] = df["NetPrice"] > df["UnitPrice"]
    report["Net fiyatı birim fiyattan yüksek (işaretlendi)"] = int(df["PriceAnomaly"].sum())

    df["Category"] = df["Category"].fillna("Unknown")
    city_map = df.dropna(subset=["City"]).groupby("CustomerID")["City"].agg(lambda s: s.mode().iloc[0])
    df["City"] = df["City"].fillna(df["CustomerID"].map(city_map)).fillna("Unknown")

    df["Revenue"] = df["Quantity"] * df["NetPrice"]          
    df["ListValue"] = df["Quantity"] * df["UnitPrice"]
    df["Discount"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    for col in ["InvoiceNo", "CustomerID", "Branch", "Category", "City", "DocType", "CustomerType"]:
        df[col] = df[col].astype(str)
    df = df.reset_index(drop=True)

    report["Temiz satır"] = len(df)
    report["Müşteri"] = df["CustomerID"].nunique()
    report["Ürün"] = df["StockCode"].nunique()
    report["Kategori"] = df["Category"].nunique()
    report["Ciro (₺)"] = float(df["Revenue"].sum())
    return df, report


# ---------------------------------------------------------------------------
STORED_COLS = ["InvoiceNo", "CustomerID", "Branch", "InvoiceDate", "StockCode", "Description", "Category", "City",
               "Quantity", "NetPrice", "UnitPrice", "DocType", "CustomerType", "IsDuplicate", "DescriptionSuspect",
               "PriceAnomaly"]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["ListValue"] = df["Quantity"] * df["UnitPrice"]
    df["Discount"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M")
    return df


def save_clean(df: pd.DataFrame, path: Path = config.CLEAN_FILE) -> Path:
    """Temiz veriyi taşınabilir sıkıştırılmış CSV olarak kaydeder (her pandas sürümünde okunur)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[STORED_COLS].copy()
    out["InvoiceDate"] = out["InvoiceDate"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, compression="gzip")
    return path


def read_clean(path: Path) -> pd.DataFrame:
    str_cols = ["InvoiceNo", "CustomerID", "Branch", "Description", "Category", "City", "DocType", "CustomerType"]
    df = pd.read_csv(path, dtype={**{c: str for c in str_cols}, "StockCode": "int64"},
                     parse_dates=["InvoiceDate"], keep_default_na=False, na_values=[""])
    df["Description"] = df["Description"].fillna("")
    for c in ["IsDuplicate", "DescriptionSuspect", "PriceAnomaly"]:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].astype(str).str.lower().eq("true")
    return _derive(df)


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    """Temiz veriyi okur. Dosya yoksa ham Excel'den üretir ve (yazılabiliyorsa) kaydeder."""
    if path.exists():
        return _add_fast_keys(read_clean(path))
    df, _ = clean(load_raw())
    try:
        save_clean(df, path)
    except OSError:
        pass
    return _add_fast_keys(df)


def _add_fast_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Hızlı groupby/nunique için tamsayı anahtarlar."""
    df["InvoiceID"] = pd.factorize(df["InvoiceNo"])[0].astype("int32")
    return df


def kpis(df: pd.DataFrame) -> dict:
    """Executive özet metrikleri."""
    return {
        "revenue": float(df["Revenue"].sum()),
        "quantity": float(df["Quantity"].sum()),
        "invoices": int(df["InvoiceNo"].nunique()),
        "customers": int(df["CustomerID"].nunique()),
        "products": int(df["StockCode"].nunique()),
        "categories": int(df["Category"].nunique()),
        "avg_discount": float((1 - df["Revenue"].sum() / df["ListValue"].sum())),
        "date_min": df["InvoiceDate"].min().date().isoformat(),
        "date_max": df["InvoiceDate"].max().date().isoformat(),
    }


def product_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """StockCode -> Description, Category (en sık değer)."""
    return (df.groupby("StockCode")
              .agg(Description=("Description", "first"), Category=("Category", "first")))
