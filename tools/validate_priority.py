"""Öncelik etiketlerinin (Hemen Ara / Teklif Sun / Takip Et) geriye dönük doğrulaması.

Model Ocak–Eylül verisiyle kurulur; her etiketteki öneriler için müşterinin ürünü Ekim–Aralık'ta
gerçekten alma oranı ölçülür. Sonuç data/backtest_results.json içine "priority" olarak eklenir.

    python tools/validate_priority.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bookpulse import config, data, next_calls 
from bookpulse.portfolio import build_portfolio

df = data.load_clean()
cut = pd.Timestamp("2020-09-30")
train, test = df[df["InvoiceDate"] <= cut], df[df["InvoiceDate"] > cut]
p, _ = build_portfolio(train)
eng = next_calls.NextCallsEngine(train, p)
pool = p[p["DNA"].isin(["Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep"]) & (p["Customers"] >= 10)]
sample = np.random.default_rng(42).choice(pool.index.to_numpy(), size=min(300, len(pool)), replace=False)
b = eng.batch(sample, n=10)
pairs = set(zip(test["CustomerID"], test["StockCode"]))
b["Hit"] = [(c, s) in pairs for c, s in zip(b["CustomerID"], b["StockCode"])]
t = b.groupby("Priority")["Hit"].agg(["mean", "size"]).rename(columns={"mean": "Isabet_%", "size": "Oneri"})
t["Isabet_%"] = (100 * t["Isabet_%"]).round(1)
print(t)
out_path = config.DATA_DIR / "backtest_results.json"
d = json.loads(out_path.read_text())
d["priority"] = [{"Öncelik": k, "İsabet %": float(r["Isabet_%"]), "Öneri": int(r["Oneri"])} for k, r in t.iterrows()]
out_path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
