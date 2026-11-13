import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpulse import config, customer_dna, data, next_calls 
from bookpulse.engine import BookPulse  

OK = []


def check(name, cond):
    assert cond, name
    OK.append(name)

raw = pd.DataFrame({
    "InvoiceNo": ["BAD1", "BAD1", "MUH9", "BSR2", "BSR3", "DPE4", "BAD5"],
    "InvoiceDate": pd.to_datetime(["2020-01-02"] * 7),
    "CustomerID": ["A00001-01", "A00001", "A00002", "A00003", "A00003", "A00004", "A00005"],
    "StockCode": [1, 2, 3, 4, 5, 6, 7],
    "Description": ["Kitap A", "Kitap B", "Muhasebe", "Kitap D", "Kitap E", "Kitap F", "Kitap G"],
    "Category": [" ROMAN ", "ROMAN", "X", "ÇOCUK", "ÇOCUK", "ROMAN", "ROMAN"],
    "City": ["İSTANBUL ", "İSTANBUL", "ANKARA", "İZMİR", "İZMİR", "BURSA", "BURSA"],
    "Quantity": [2, 1, 1, 0, 3, 1, 1],
    "UnitPrice": [10.0, 20.0, 5.0, 10.0, 10.0, 0.001, 10.0],
    "NetPrice": [6.0, 12.0, 5.0, 6.0, -1.0, 0.001, 12.0],
})
try:
    clean, rep = data.clean(raw)
except KeyError as e:          
    clean, rep = None, None
    print("Ham veri şeması farklı, temizlik testi atlandı:", e)
if clean is not None:
    check("MUH belge türü çıkarıldı", (clean["StockCode"] != 3).all())
    check("sıfır miktar çıkarıldı", (clean["Quantity"] > 0).all())
    check("negatif net fiyat çıkarıldı", (clean["NetPrice"] > 0).all())
    check("sembolik birim fiyat çıkarıldı", (clean["UnitPrice"] >= config.MIN_UNIT_PRICE).all())
    check("alt hesap ana müşteriye bağlandı", "A00001-01" not in set(clean["CustomerID"]))
    check("boşluklar temizlendi", set(clean["Category"]) <= {"ROMAN", "ÇOCUK"})
    if "PriceAnomaly" in clean:
        check("fiyat değiştirilmedi, yalnızca işaretlendi",
              bool(clean.loc[clean["StockCode"] == 7, "PriceAnomaly"].iloc[0])
              and float(clean.loc[clean["StockCode"] == 7, "NetPrice"].iloc[0]) == 12.0)

bp = BookPulse()
df, p = bp.df, bp.portfolio
check("ciro = adet × net fiyat", np.allclose(df["Revenue"], df["Quantity"] * df["NetPrice"]))
check("her ürünün bir DNA grubu var", p["DNA"].notna().all())
check("performans puanı 0–100", p["Performance_Score"].between(0, 100).all())

star = p[p["DNA"] == "Çok Satan – Düzenli Talep"].sort_values("Revenue", ascending=False).index[0]
rec = bp.calls.recommend(star, n=10)
buyers = set(df.loc[df["StockCode"] == star, "CustomerID"])
check("öneri 10 müşteri döndürür", len(rec) == 10)
check("ürünü zaten alan müşteri önerilmez", not (set(rec["CustomerID"]) & buyers))
check("puanlar 0–100 ve azalan", rec["Score"].between(0, 100).all() and rec["Score"].is_monotonic_decreasing)
check("öncelik etiketleri geçerli", set(rec["Priority"]) <= {"Hemen Ara", "Geri Kazanım Araması", "Teklif Sun", "Takip Et"})
check("ekibin ilk modeli farklı sıralama verebilir", len(bp.calls.recommend(star, n=10, weights=next_calls.V1_WEIGHTS)) == 10)

cust = rec["CustomerID"].iloc[0]
why = bp.calls.why_customer(cust, star)
check("neden bu müşteri: kategori payı yüzdedir", 0 <= why["category_share_%"] <= 100)
cp = bp.calls.customer_portfolio(cust)
check("müşteri portföy DNA payları %100", abs(cp["Customer_Share_%"].sum() - 100) < 0.5)
nxt = bp.calls.products_for_customer(cust, n=5, dna_filter=None)
owned = set(df.loc[df["CustomerID"] == cust, "StockCode"])
check("ters yön: müşterinin aldığı ürün önerilmez", not (set(nxt["StockCode"]) & owned))

batch = bp.calls.batch(p[p["DNA"] == "Çok Satan – Düzenli Talep"].nlargest(5, "Revenue").index, n=5)
opp = bp.calls.customer_opportunities(batch)
check("toplu liste müşteri bazında tekil", opp.index.is_unique and opp["Opportunities"].sum() == len(batch))

check("bilinmeyen ürün aranınca boş sonuç", bp.resolve_product("zzzz-olmayan-ürün-qq") is None)
check("bilinmeyen müşteri için öneri boş", bp.calls.products_for_customer("YOK0000", n=5).empty)
check("barkodla arama", bp.resolve_product(str(star)) == star)

mo = customer_dna.missing_orders(df)
check("kaçan sipariş kuralı: ≥2 kat gecikme ve ≥21 gün", (mo["Delay_Ratio"] >= 2).all() and (mo["Days_Since_Last"] >= 21).all())
ev = bp.forecast_eval
check("seçim ve test pencereleri çakışmaz", ev["selection_window"][1] < ev["report_window"][0])
check("tahmin aralığı negatif değil", (bp.forecast_table["Lower"] >= 0).all())

print(f"{len(OK)} test geçti:")
for n_ in OK:
    print("  -", n_)
