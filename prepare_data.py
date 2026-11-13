"""Ham Excel'i temizleyip data/transactions_clean.csv.gz olarak kaydeder.
"""
import json
import sys
from pathlib import Path

from bookpulse import config, data


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RAW_XLSX
    print(f"Okunuyor: {src}")
    raw = data.load_raw(src)
    df, report = data.clean(raw)
    out = data.save_clean(df)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
