"""Sıradaki 10 Arama ağırlıklarının nasıl seçildiğini yeniden üretir (sızıntısız doğrulama).

Doğrulama: model Ocak–Haziran verisiyle kurulur, öneriler Temmuz–Eylül alımlarıyla ölçülür.
Ekim–Aralık test penceresine bu aşamada hiç bakılmaz (nihai test: tools/precompute_backtest.py).

Sonuç (150 ürün, seed 7): benzer ürün sinyali ağırlıklı adaylar (%25 / %35 / %50: 8,40 / 8,07 / 8,27)
ekibin ilk modelini (7,33) ve eşit ağırlıkları (7,60) geçiyor. Bu üç aday arasındaki fark örneklem
gürültüsü düzeyinde (60 üründe sıralama değişiyor), bu yüzden ortadaki %35 kullanıldı.

    python tools/tune_weights.py            # varsayılan: 150 ürün
    python tools/tune_weights.py 50         # hızlı deneme
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bookpulse import config, data, next_calls  # noqa: E402

N_PRODUCTS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
KEYS = ["category_qty", "category_share", "frequency", "diversity", "similar_items", "recency"]

CANDIDATES = {
    "ekibin ilk modeli (v1)": next_calls.V1_WEIGHTS,
    "eşit": {k: 1 / 6 for k in KEYS},
    "kategori ağırlıklı": dict(category_qty=.35, category_share=.15, frequency=.20, diversity=.10, similar_items=.10, recency=.10),
    "sıklık ağırlıklı": dict(category_qty=.15, category_share=.10, frequency=.40, diversity=.10, similar_items=.15, recency=.10),
    "benzer ürün %25": dict(category_qty=.20, category_share=.10, frequency=.20, diversity=.15, similar_items=.25, recency=.10),
    "benzer ürün %35 (kullanılan)": config.NEXT_CALLS_WEIGHTS,
    "benzer ürün %50": dict(category_qty=.10, category_share=.05, frequency=.15, diversity=.10, similar_items=.50, recency=.10),
}

df = data.load_clean()
df = df[df["InvoiceDate"] <= "2020-09-30"]     
rows = []
for name, w in CANDIDATES.items():
    res = next_calls.backtest(df, cutoff="2020-06-30", horizon_end="2020-09-30", n_products=N_PRODUCTS, seed=7,
                              weights=w)
    t = res["table"]
    p10 = t.iloc[0]["Precision@10_%"]
    rows.append({"Aday": name, "Doğrulama İsabet@10 (%)": float(p10)})
    print(f"{name:28s} {p10:6.2f}")

best = max(rows, key=lambda r: r["Doğrulama İsabet@10 (%)"])
print("\nEn iyi aday:", best["Aday"])
(config.DATA_DIR / "weight_tuning.json").write_text(json.dumps(
    {"train": "2020-01-01..2020-06-30", "validation": "2020-07-01..2020-09-30", "n_products": N_PRODUCTS, "seed": 7,
     "results": rows, "best": best["Aday"]}, ensure_ascii=False, indent=1))
