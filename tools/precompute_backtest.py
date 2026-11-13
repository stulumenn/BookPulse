"""Sıradaki 10 Arama geriye dönük testini önceden hesaplayıp data/backtest_results.json olarak kaydeder.
Pano bu dosyayı okuyarak sonucu anında gösterir (canlıda 30 sn beklemek gerekmez).

    python tools/precompute_backtest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bookpulse import config, data, next_calls  # noqa: E402

res = next_calls.backtest(data.load_clean(), n_products=300, seed=42)
out = {"cutoff": res["cutoff"], "n_products": res["n_products"], "seed": 42,
       "weights": config.NEXT_CALLS_WEIGHTS, "v1_weights": next_calls.V1_WEIGHTS,
       "note": "Ağırlıklar Oca–Haz → Tem–Eyl doğrulama penceresinde seçildi; sonuç Oca–Eyl → Eki–Ara testinden.",
       "table": res["table"].reset_index().rename(columns={"index": "Model"}).to_dict("records")}
(config.DATA_DIR / "backtest_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
