"""Yapay Zekâ Satış Danışmanı – Ollama dil modeli + araç çağırma (tool calling) ile kontrollü veri sorgulama.

Chatbot ham veriye serbestçe erişmez; yalnızca aşağıdaki analiz fonksiyonlarını (tool) çağırabilir.
Böylece cevaplar gerçek analiz sonuçlarına dayanır ve uydurma sayı riski azalır.

Bağlantı (ortam değişkenleri veya Streamlit Cloud > Settings > Secrets):
    OLLAMA_HOST     varsayılan http://localhost:11434 (yerel Ollama)
    OLLAMA_API_KEY  verilirse Ollama Cloud (https://ollama.com) kullanılır
    OLLAMA_MODEL    varsayılan qwen2.5:7b (yerel)
Ollama'ya ulaşılamazsa (ör. Streamlit Community Cloud'da yerel Ollama yoktur) anahtar kelime
tabanlı yedek mod devreye girer: soru doğru analize yönlendirilir, tablo ve grafik yine gösterilir.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import basket, config
from .engine import BookPulse

MAX_ROWS = 25


# ---------------------------------------------------------------------------
def _df_payload(df: pd.DataFrame, n: int = MAX_ROWS) -> dict:
    d = df.head(n).copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(2)
    d = d.replace({np.nan: None})
    return {"columns": [str(c) for c in d.columns], "rows": d.astype(object).values.tolist(), "total_rows": int(len(df))}


TOOLS = [
    {"name": "get_kpis", "description": "Genel satış özeti: ciro, adet, fatura, müşteri, ürün sayısı, ortalama iskonto, tarih aralığı. İsteğe bağlı tarih filtresi.",
     "input_schema": {"type": "object", "properties": {
         "start_date": {"type": "string", "description": "YYYY-MM-DD"}, "end_date": {"type": "string", "description": "YYYY-MM-DD"}}}},
    {"name": "top_entities", "description": "En iyi müşteri / ürün / kategori / şehirleri ciro veya adete göre sıralar. Tarih filtresi opsiyonel.",
     "input_schema": {"type": "object", "properties": {
         "dimension": {"type": "string", "enum": ["customer", "product", "category", "city"]},
         "metric": {"type": "string", "enum": ["revenue", "quantity"], "default": "revenue"},
         "n": {"type": "integer", "default": 10},
         "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["dimension"]}},
    {"name": "category_trend", "description": "Son N ayı önceki N ayla karşılaştırarak satışları en çok düşen/artan kategorileri bulur.",
     "input_schema": {"type": "object", "properties": {
         "months": {"type": "integer", "default": 3}, "direction": {"type": "string", "enum": ["down", "up"], "default": "down"},
         "n": {"type": "integer", "default": 10}, "min_quantity": {"type": "integer", "default": 500}}}},
    {"name": "monthly_sales", "description": "Aylık ciro ve adet zaman serisi; opsiyonel olarak tek bir kategori için.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"}}}},
    {"name": "portfolio_products", "description": "Portföy DNA / optimizasyon tablosundan ürün listesi. DNA grubu ve/veya aksiyon ile filtrelenir.",
     "input_schema": {"type": "object", "properties": {
         "dna": {"type": "string", "enum": ["Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep", "Az Satan – Düzenli Talep", "Az Satan – Değişken Talep", "Yetersiz Satış Geçmişi"]},
         "action": {"type": "string", "description": "Örn. 'Sürekli Bulunurluğu Değerlendir', 'Ürün Performansını İncele', 'İzle (Düşüşte)'"},
         "category": {"type": "string"},
         "sort_by": {"type": "string", "enum": ["Revenue", "Total_Qty", "Performance_Score", "Trend"], "default": "Revenue"},
         "n": {"type": "integer", "default": 10}}}},
    {"name": "portfolio_summary", "description": "DNA gruplarının ve önerilen aksiyonların ürün sayısı ve ciro payları, kullanılan eşikler.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "search_product", "description": "Ürün adında arama yapar ve StockCode döndürür. Diğer ürün araçlarından önce ürün kodunu bulmak için kullan.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}}, "required": ["query"]}},
    {"name": "next_calls_for_product", "description": "Bir ürün için aranması gereken en uygun müşteriler (puan, öncelik, gerekçe). product: StockCode veya ürün adı.",
     "input_schema": {"type": "object", "properties": {"product": {"type": "string"}, "n": {"type": "integer", "default": 10}}, "required": ["product"]}},
    {"name": "top_call_list", "description": "Tüm Çok Satan – Düzenli Talep ve Çok Satan – Değişken Talep ürünlerinin en yüksek ciroluları için birleşik arama listesi: en yüksek potansiyelli müşteri-ürün eşleşmeleri.",
     "input_schema": {"type": "object", "properties": {"n_products": {"type": "integer", "default": 30}, "n": {"type": "integer", "default": 10},
                                                       "priority": {"type": "string", "enum": ["Hemen Ara", "Teklif Sun", "Takip Et", "Geri Kazanım Araması"]}}}},
    {"name": "products_for_customer", "description": "Bir müşteriye önerilecek ürünler (benzer müşterilerin aldıkları, henüz almadıkları).",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}, "n": {"type": "integer", "default": 10}}, "required": ["customer_id"]}},
    {"name": "customer_profile", "description": "Tek müşterinin RFM değerleri, segmenti, kümesi, en çok aldığı kategoriler ve önerilen aksiyon.",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}},
    {"name": "customer_segments", "description": "Müşteri DNA segment özeti; segment verilirse o segmentteki müşteriler (cirosuna göre).",
     "input_schema": {"type": "object", "properties": {
         "segment": {"type": "string", "enum": ["En Değerli Müşteriler", "Sadık Müşteriler", "Gelişen Müşteriler", "Yeni Gözlenen Müşteriler", "İlgi Gerektiren Müşteriler", "Riskli Müşteriler", "Riskli Değerli Müşteriler", "Pasif Müşteriler"]},
         "n": {"type": "integer", "default": 15}}}},
    {"name": "cross_sell", "description": "Bir ürünü (veya kategori) alanların birlikte aldığı ürünler (Sepet Analizi: support, confidence, lift).",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string", "description": "Ürün adı/StockCode ya da kategori adı"},
                                                       "level": {"type": "string", "enum": ["product", "category"], "default": "product"},
                                                       "n": {"type": "integer", "default": 10}}, "required": ["item"]}},
    {"name": "top_bundles", "description": "En güçlü birlikte satın alma çiftleri (paket önerisi).",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "default": 10},
                                                       "cross_category_only": {"type": "boolean", "default": False},
                                                       "level": {"type": "string", "enum": ["product", "category"], "default": "product"}}}},
    {"name": "customer_portfolio_dna", "description": "Müşterinin Portföy DNA'sı: alımlarının ürün DNA gruplarına dağılımı ve şirket geneliyle farkı.",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}},
    {"name": "why_customer", "description": "Bir müşterinin bir ürün için neden önerildiğinin kanıtları (kategori payı, benzer ürün alımı, son alışveriş).",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}, "product": {"type": "string"}},
                      "required": ["customer_id", "product"]}},
    {"name": "missing_orders", "description": "Kaçan Sipariş Dedektörü: normal sipariş aralığının en az 2 katı süredir sipariş vermeyen müşteriler (ciroya göre).",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}, "city": {"type": "string"}}}},
    {"name": "demand_forecast", "description": "Kategoriler için önümüzdeki haftaların talep (adet) tahmini, değişim yüzdesi ve model doğruluğu.",
     "input_schema": {"type": "object", "properties": {"weeks": {"type": "integer", "default": 4}, "category": {"type": "string"}}}},
]


# ---------------------------------------------------------------------------
class ToolRunner:
    def __init__(self, bp: BookPulse):
        self.bp = bp

    def run(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"Bilinmeyen araç: {name}"}
        try:
            return fn(**(args or {}))
        except Exception as e:     
            return {"error": f"{type(e).__name__}: {e}"}

    def _filter(self, start_date=None, end_date=None):
        d = self.bp.df
        if start_date:
            d = d[d["InvoiceDate"] >= pd.Timestamp(start_date)]
        if end_date:
            d = d[d["InvoiceDate"] <= pd.Timestamp(end_date)]
        return d

    def t_get_kpis(self, start_date=None, end_date=None):
        from .data import kpis
        d = self._filter(start_date, end_date)
        if d.empty:
            return {"error": "Bu tarih aralığında veri yok (veri 2020-01-02 – 2020-12-31)."}
        k = kpis(d)
        k["note"] = "Ciro = Adet × Net Fiyat (hesaplanan satış tutarı, TL)."
        return k

    def t_top_entities(self, dimension, metric="revenue", n=10, start_date=None, end_date=None):
        d = self._filter(start_date, end_date)
        col = {"customer": "CustomerID", "product": "StockCode", "category": "Category", "city": "City"}[dimension]
        val = "Revenue" if metric == "revenue" else "Quantity"
        t = d.groupby(col).agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"),
                               Orders=("InvoiceID", "nunique")).sort_values(val, ascending=False)
        t["Share_%"] = 100 * t[val] / t[val].sum()
        if dimension == "product":
            t.insert(0, "Product", t.index.map(self.bp.products["Description"]))
        if dimension == "customer":
            t = t.join(self.bp.customers[["Segment", "City"]])
        return {"table": _df_payload(t.reset_index(), n)}

    def t_category_trend(self, months=3, direction="down", n=10, min_quantity=500):
        m = self.bp.df.groupby(["Category", "Month"])["Quantity"].sum().unstack(fill_value=0)
        last, prev = m.iloc[:, -months:].sum(axis=1), m.iloc[:, -2 * months:-months].sum(axis=1)
        t = pd.DataFrame({"Previous": prev, "Last": last})
        t = t[(t["Previous"] + t["Last"]) >= min_quantity]
        t["Change_%"] = 100 * (t["Last"] - t["Previous"]) / t["Previous"].replace(0, np.nan)
        t = t.sort_values("Change_%", ascending=(direction == "down"))
        periods = list(m.columns.astype(str))
        return {"table": _df_payload(t.reset_index(), n),
                "periods": {"previous": periods[-2 * months:-months], "last": periods[-months:]}}

    def t_monthly_sales(self, category=None):
        d = self.bp.df if not category else self.bp.df[self.bp.df["Category"].str.lower() == category.lower()]
        if d.empty:
            return {"error": f"Kategori bulunamadı: {category}"}
        t = d.groupby("Month").agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum")).reset_index()
        t["Month"] = t["Month"].astype(str)
        return {"table": _df_payload(t, 12), "chart": {"type": "line", "x": "Month", "y": "Revenue"}}

    def t_portfolio_products(self, dna=None, action=None, category=None, sort_by="Revenue", n=10):
        p = self.bp.portfolio
        if dna:
            p = p[p["DNA"] == dna]
        if action:
            p = p[p["Action"].str.contains(action, case=False, regex=False)]
        if category:
            p = p[p["Category"].str.contains(category, case=False, regex=False)]
        cols = ["Description", "Category", "DNA", "Action", "Total_Qty", "Revenue", "CV", "Trend",
                "Customers", "Performance_Score", "Peak_Month"]
        t = p.sort_values(sort_by, ascending=False)[cols].reset_index()
        return {"table": _df_payload(t, n)}

    def t_portfolio_summary(self):
        from .portfolio import action_summary, dna_summary
        return {"dna": _df_payload(dna_summary(self.bp.portfolio).reset_index()),
                "actions": _df_payload(action_summary(self.bp.portfolio).reset_index()),
                "thresholds": self.bp.thresholds}

    def t_search_product(self, query, n=5):
        t = self.bp.find_products(query, n)[["Description", "Category", "DNA", "Revenue", "Total_Qty"]]
        return {"table": _df_payload(t.reset_index(), n)} if len(t) else {"error": f"'{query}' bulunamadı."}

    def t_next_calls_for_product(self, product, n=10):
        sc = self.bp.resolve_product(product)
        if sc is None:
            return {"error": f"Ürün bulunamadı: {product}"}
        r = self.bp.calls.recommend(sc, n=n)
        return {"product": self.bp.products.loc[sc, "Description"], "stock_code": sc,
                "dna": self.bp.portfolio.loc[sc, "DNA"], "table": _df_payload(r.drop(columns=["StockCode", "Product"]), n)}

    def t_top_call_list(self, n_products=30, n=10, priority=None):
        p = self.bp.portfolio
        top = p[p["DNA"].isin(["Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep"])].nlargest(n_products, "Revenue").index
        b = self.bp.calls.batch(top, n=5)
        if priority:
            b = b[b["Priority"] == priority]
        b = b.sort_values("Score", ascending=False).drop_duplicates("CustomerID")
        return {"table": _df_payload(b[["CustomerID", "Product", "DNA", "Score", "Priority", "Reason"]], n)}

    def t_products_for_customer(self, customer_id, n=10):
        r = self.bp.calls.products_for_customer(customer_id.upper().strip(), n=n)
        return {"table": _df_payload(r, n)} if len(r) else {"error": f"Müşteri bulunamadı: {customer_id}"}

    def t_customer_profile(self, customer_id):
        cid = customer_id.upper().strip()
        c = self.bp.customers
        if cid not in c.index:
            return {"error": f"Müşteri bulunamadı: {cid}"}
        row = c.loc[cid]
        d = self.bp.df[self.bp.df["CustomerID"] == cid]
        cats = d.groupby("Category")["Revenue"].sum().nlargest(5)
        return {
            "customer_id": cid, "city": row["City"], "segment": row["Segment"], "cluster": row["Cluster_Name"],
            "recency_days": int(row["Recency"]), "orders": int(row["Frequency"]), "revenue": round(float(row["Monetary"]), 2),
            "avg_order_value": round(float(row["Avg_Order_Value"]), 2), "categories_bought": int(row["Categories"]),
            "rfm_score": row["RFM_Score"], "action": row["Action"],
            "top_categories": {k: round(float(v), 2) for k, v in cats.items()},
            "branches": sorted(d["Branch"].unique().tolist()),
        }

    def t_customer_segments(self, segment=None, n=15):
        from .customer_dna import segment_summary
        c = self.bp.customers
        if not segment:
            return {"table": _df_payload(segment_summary(c).reset_index())}
        t = c[c["Segment"] == segment].sort_values("Monetary", ascending=False)
        return {"segment": segment, "table": _df_payload(
            t[["City", "Recency", "Frequency", "Monetary", "Categories", "Cluster_Name", "Action"]].reset_index(), n)}

    def t_cross_sell(self, item, level="product", n=10):
        if level == "category":
            cats = self.bp.category_rules
            match = [c for c in cats["A"].unique() if item.lower() in str(c).lower()]
            if not match:
                return {"error": f"Kategori bulunamadı: {item}"}
            r = basket.cross_sell(cats, match[0], n)
            return {"item": match[0], "table": _df_payload(r[["B", "Count_AB", "Support", "Confidence", "Lift"]], n)}
        sc = self.bp.resolve_product(item)
        if sc is None:
            return {"error": f"Ürün bulunamadı: {item}"}
        r = basket.cross_sell(self.bp.product_rules, sc, n)
        if r.empty:
            return {"product": self.bp.products.loc[sc, "Description"], "message": "Bu ürün için anlamlı birliktelik kuralı yok (az sepette görülüyor)."}
        return {"product": self.bp.products.loc[sc, "Description"],
                "table": _df_payload(r[["B", "B_Name", "B_Category", "Count_AB", "Confidence", "Lift"]], n)}

    def t_top_bundles(self, n=10, cross_category_only=False, level="product"):
        rules = self.bp.product_rules if level == "product" else self.bp.category_rules
        b = basket.bundles(rules, n, cross_category_only=cross_category_only)
        cols = ["A_Name", "B_Name", "Count_AB", "Confidence", "Lift"] if level == "product" else ["A", "B", "Count_AB", "Confidence", "Lift"]
        return {"table": _df_payload(b[cols], n)}

    def t_customer_portfolio_dna(self, customer_id):
        cid = customer_id.upper().strip()
        t = self.bp.calls.customer_portfolio(cid)
        return {"customer_id": cid, "table": _df_payload(t.reset_index())} if len(t) else {"error": f"Müşteri bulunamadı: {cid}"}

    def t_why_customer(self, customer_id, product):
        sc = self.bp.resolve_product(product)
        if sc is None:
            return {"error": f"Ürün bulunamadı: {product}"}
        cid = customer_id.upper().strip()
        if cid not in self.bp.calls.c_idx.index:
            return {"error": f"Müşteri bulunamadı: {cid}"}
        out = self.bp.calls.why_customer(cid, sc)
        out["product"] = self.bp.products.loc[sc, "Description"]
        return out

    def t_missing_orders(self, n=10, city=None):
        from .customer_dna import missing_orders
        t = missing_orders(self.bp.df)
        if city:
            t = t[t["City"].str.upper() == city.upper()]
        return {"table": _df_payload(t.reset_index(), n), "total": int(len(t)),
                "note": "Tipik sipariş aralığının en az 2 katı (ve 21+ gün) sessiz kalan müşteriler; risk sinyalidir."}

    def t_demand_forecast(self, weeks=4, category=None):
        from .forecast import outlook
        fc = self.bp.forecast_table
        ev = self.bp.forecast_eval
        if category:
            match = [c for c in fc["Series"].unique() if category.lower() in str(c).lower()]
            if not match:
                return {"error": f"Tahmin edilen kategoriler arasında yok: {category}. Mevcut: {sorted(fc['Series'].unique().tolist())}"}
            t = fc[fc["Series"] == match[0]].head(weeks)
            return {"category": match[0], "table": _df_payload(t), "validation_wape_%": ev["per_series"].loc[match[0]].to_dict()}
        return {"table": _df_payload(outlook(self.bp.weekly, fc, weeks).rename_axis("Series").reset_index()),
                "validation": ev["overall"].reset_index().rename(columns={"index": "Method"}).to_dict("records"),
                "note": "Veri yalnızca 2020'yi kapsıyor; tahminler kısa vadeli ve en iyi doğrulanan yönteme göre seçildi."}


# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Sen BookPulse'un Yapay Zekâ Satış Danışmanısın: bir kitap/kırtasiye B2B distribütörünün satış ekibine ve yöneticilerine yardım eden bir analistsin.
Kurallar:
- Sayıları ASLA tahmin etme veya uydurma; her sayısal iddia için araçları (tools) kullan.
- Cevaplarını Türkçe, kısa ve aksiyon odaklı ver: önce cevap, sonra 2-4 maddelik öneri.
- Veri 2 Ocak – 31 Aralık 2020 arasını kapsar (Covid yılı; Nisan–Temmuz düşüşü normal değil, kapanma etkisi).
- "Ciro" = Adet × Net Fiyat ile hesaplanan satış tutarıdır; fiyat alanlarının ticari tanımı doğrulanmadı, gerekirse belirt.
- Ürün adı verilirse önce search_product ile doğru ürünü bul.
- Tablo sonuçlarını özetle; uygulamada tablo zaten kullanıcıya gösteriliyor, satır satır tekrar etme.
- İngilizce terim kullanma; teknik bir terim (ör. kaldıraç, RFM, değişkenlik) geçerse yanında kısa bir açıklama ver.
Modüller: Portföy DNA/Optimizasyonu, Sıradaki 10 Arama, Müşteri DNA (RFM+K-Means), Sepet Analizi, Talep Tahmini."""


@dataclass
class AdvisorReply:
    text: str
    tool_calls: list = field(default_factory=list)    
    error: str | None = None
    mode: str = "claude"


def _clean_schema(schema):
    """Araç şemasını sadeleştirir ('default' alanları bazı modellerde sorun çıkarır)."""
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != "default"}
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


OLLAMA_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": _clean_schema(t["input_schema"])}} for t in TOOLS]


def _dump(out) -> str:
    return json.dumps(out, ensure_ascii=False, default=str)[:12000]


# ---------------------------------------------------------------------------
class OllamaError(Exception):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status
        self.body = body


def ollama_settings() -> dict:
    """Ayarları ortam değişkenlerinden okur (Streamlit Cloud'da 'Secrets' da ortam değişkeni olarak gelir)."""
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    host = os.getenv("OLLAMA_HOST", "").strip() or (config.OLLAMA_CLOUD_HOST if key else config.OLLAMA_LOCAL_HOST)
    if not host.startswith("http"):
        host = "http://" + host
    model = os.getenv("OLLAMA_MODEL", "").strip() or (config.OLLAMA_CLOUD_MODEL if key else config.OLLAMA_LOCAL_MODEL)
    return {"host": host.rstrip("/"), "api_key": key, "model": model}


def _strip_none(obj):
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def _request(host: str, path: str, api_key: str = "", payload: dict | None = None, timeout: float = 180):
    """Ollama'ya istek. 'ollama' Python paketi (requirements.txt) varsa onu, yoksa düz HTTP kullanır.
    Her iki durumda da Ollama REST API'siyle aynı biçimde sözlük döndürür."""
    try:
        import ollama as _ollama
    except ImportError:
        _ollama = None

    if _ollama is not None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            client = _ollama.Client(host=host, headers=headers, timeout=timeout)
            if path == "/api/tags":
                r = client.list()
            elif path == "/api/chat":
                r = client.chat(**payload)
            else:
                raise ValueError(path)
            data = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            return _strip_none(data)
        except _ollama.ResponseError as e:
            raise OllamaError(f"HTTP {e.status_code}: {e.error}", e.status_code, str(e.error)) from None
        except OllamaError:
            raise
        except Exception as e:                                
            raise OllamaError(f"Bağlantı hatası: {e}") from None

    import urllib.error
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(host.rstrip("/") + path, data=data, headers=headers,
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise OllamaError(f"HTTP {e.code}: {body[:400]}", e.code, body) from None
    except Exception as e:
        raise OllamaError(f"Bağlantı hatası: {e}") from None


def ollama_models(host: str | None = None, api_key: str | None = None, timeout: float = 1.5) -> list[str] | None:
    """Ollama'ya ulaşılabiliyorsa model adlarını, ulaşılamıyorsa None döndürür."""
    st_ = ollama_settings()
    try:
        data = _request(host or st_["host"], "/api/tags", api_key if api_key is not None else st_["api_key"],
                        timeout=timeout)
        return sorted(m.get("name") or m.get("model") for m in data.get("models", []))
    except OllamaError:
        return None


def ollama_status(host: str | None = None, api_key: str | None = None, model: str | None = None) -> dict:
    """Ollama'nın durumunu adım adım kontrol eder.

    step: 'ready'      -> çalışıyor ve önerilen/istenen model kurulu
          'no_model'   -> çalışıyor ama hiç model yok
          'not_running'-> bilgisayarda kurulu ama çalışmıyor
          'not_installed' -> bu bilgisayarda Ollama bulunamadı
    """
    import importlib.util
    import shutil
    s = ollama_settings()
    host = (host or s["host"]).rstrip("/")
    api_key = s["api_key"] if api_key is None else api_key
    model = model or s["model"]
    models = ollama_models(host, api_key)
    status = {"host": host, "model": model, "models": models or [], "cloud": bool(api_key),
              "python_package": importlib.util.find_spec("ollama") is not None,
              "app_installed": shutil.which("ollama") is not None}
    if models is None:
        status["step"] = "not_running" if status["app_installed"] else "not_installed"
    elif not models:
        status["step"] = "no_model"
    else:
        status["step"] = "ready"
        status["model"] = model if model in models else models[0]
    return status


def setup_guide(status: dict) -> str:
    """Ollama kurulum rehberi (Markdown). Kullanıcının takıldığı adım işaretlenir."""
    m = config.OLLAMA_LOCAL_MODEL
    step = status.get("step")
    head = {
        "not_installed": "**Ollama bu bilgisayarda bulunamadı.** Yapay Zekâ Satış Danışmanı, dil modelini kendi bilgisayarınızda "
                         "çalıştıran ücretsiz **Ollama** uygulamasıyla çalışır. Kurulum yaklaşık 10 dakika sürer:",
        "not_running": "**Ollama kurulu ama çalışmıyor.** Ollama uygulamasını açın (veya terminalde `ollama serve` çalıştırın), "
                       "ardından aşağıdaki **Tekrar kontrol et** düğmesine basın.",
        "no_model": f"**Ollama çalışıyor ama hiç model indirilmemiş.** Terminalde `ollama pull {m}` çalıştırın.",
    }.get(step, "")
    steps = f"""
1. **Ollama'yı indirin ve kurun:** https://ollama.com/download
   - macOS / Windows: indirilen kurulum dosyasını açın.
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
2. **Ollama'yı başlatın:** uygulamayı açın (arka planda çalışır). Kontrol için terminalde `ollama --version`.
3. **Modeli indirin** (bir kez, ~4,7 GB; en az 8 GB RAM önerilir):
   ```
   ollama pull {m}
   ```
4. **Python paketlerini kurun** (Ollama'nın Python istemcisi dahildir):
   ```
   pip install -r requirements.txt
   ```
5. Bu sayfada **Tekrar kontrol et**'e basın (veya uygulamayı yeniden başlatın).

Not: Streamlit Community Cloud gibi bulut sunucularında Ollama çalışmaz; danışmanın doğal dilde cevap vermesi için
projeyi kendi bilgisayarınızda çalıştırın. Panonun diğer sayfaları her yerde çalışır."""
    return (head + "\n" + steps).strip()


def ping(host: str, api_key: str, model: str) -> str:
    """Araçsız tek mesajlık bağlantı testi."""
    r = _request(host, "/api/chat", api_key, {"model": model, "stream": False,
                                              "messages": [{"role": "user", "content": "Sadece 'Hazırım' yaz."}]},
                 timeout=120)
    return (r.get("message") or {}).get("content", "")


# ---------------------------------------------------------------------------
PROVIDERS = {
    "ollama": {"label": "Ollama (yerel, ücretsiz)", "kind": "ollama", "key_env": "OLLAMA_API_KEY",
               "model_env": "OLLAMA_MODEL", "default_model": config.OLLAMA_LOCAL_MODEL,
               "help": "Bilgisayarınızda çalışan Ollama. Anahtar gerekmez."},
    "openai": {"label": "OpenAI (GPT)", "kind": "openai", "base_url": None, "key_env": "OPENAI_API_KEY",
               "model_env": "OPENAI_MODEL", "default_model": config.OPENAI_MODEL_DEFAULT,
               "help": "Anahtar: https://platform.openai.com/api-keys"},
    "claude": {"label": "Anthropic (Claude)", "kind": "claude", "key_env": "ANTHROPIC_API_KEY",
               "model_env": "ANTHROPIC_MODEL", "default_model": config.ANTHROPIC_MODEL_DEFAULT,
               "help": "Anahtar: https://console.anthropic.com/settings/keys"},
    "gemini": {"label": "Google (Gemini)", "kind": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",
               "model_env": "GEMINI_MODEL", "default_model": config.GEMINI_MODEL_DEFAULT,
               "help": "Anahtar: https://aistudio.google.com/apikey"},
}
OPENAI_TOOLS = [{"type": "function", "function": (
    {k: v for k, v in t["function"].items() if k != "parameters"}
    if not t["function"]["parameters"].get("properties") else t["function"])} for t in OLLAMA_TOOLS]


def provider_key(provider: str) -> str:
    """Ortam değişkeni / Streamlit Secrets'tan anahtar (Streamlit Cloud'da Secrets ortam değişkeni olarak gelir)."""
    return os.getenv(PROVIDERS[provider]["key_env"], "").strip()


def _extra(obj) -> dict:
    extra = getattr(obj, "model_extra", None) or {}
    return {k: v for k, v in extra.items() if v is not None}


def _openai_assistant_turn(msg) -> dict:
    """Araç çağıran model mesajını geri gönderilecek biçimde kurar. Gemini 3'ün 'thought signature' gibi
    ek alanları korunur (gönderilmezse 400 hatası verir)."""
    calls = []
    for tc in msg.tool_calls:
        d = {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
        d.update(_extra(tc))
        calls.append(d)
    turn = {"role": "assistant", "content": msg.content or None, "tool_calls": calls}
    turn.update({k: v for k, v in _extra(msg).items() if k not in turn})
    return turn


def build_context(ctx: dict | None) -> str:
    """Kullanıcının panoda o an incelediği ürün/müşteri bilgisini sistem mesajına ekler."""
    if not ctx:
        return ""
    lines = [f"- {k}: {v}" for k, v in ctx.items() if v not in (None, "", [])]
    if not lines:
        return ""
    return ("\n\nKullanıcının panoda şu an incelediği bağlam (soru 'bu müşteri', 'bu ürün' gibi ifadeler içeriyorsa "
            "bunları kullan):\n" + "\n".join(lines))


class SalesAdvisor:
    """Sağlayıcıdan bağımsız satış danışmanı: model yalnızca analiz araçlarını çağırır, hesaplamayı Python yapar.
    provider: 'ollama' | 'openai' | 'claude' | 'gemini'. Bağlantı yoksa anahtar kelime tabanlı yedek mod."""

    def __init__(self, bp: BookPulse, provider: str = "ollama", model: str | None = None, api_key: str | None = None,
                 host: str | None = None, available: bool | None = None, context: dict | None = None):
        self.runner = ToolRunner(bp)
        self.provider = provider
        spec = PROVIDERS[provider]
        self.kind = spec["kind"]
        self.context = context or {}
        s = ollama_settings()
        self.host = (host or s["host"]).rstrip("/")
        self.api_key = (api_key if api_key is not None else
                        (s["api_key"] if provider == "ollama" else provider_key(provider)))
        self.model = model or os.getenv(spec["model_env"], "").strip() or spec["default_model"]
        self.client = None
        if self.kind == "ollama":
            self.available = (ollama_models(self.host, self.api_key) is not None) if available is None else available
        else:
            self.available = bool(self.api_key) if available is None else available
            if self.available:
                try:
                    if self.kind == "openai":
                        import openai
                        self.client = openai.OpenAI(api_key=self.api_key, base_url=spec.get("base_url"))
                    else:
                        import anthropic
                        self.client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    self.available = False

    @property
    def mode(self) -> str:
        return self.provider if self.available else "fallback"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT + build_context(self.context)

    def ask(self, question: str, history: list | None = None, max_steps: int = 6) -> AdvisorReply:
        if not self.available:
            rep = fallback_answer(self.runner, question)
            rep.text = ("Dil modeline bağlı değilim; soruyu anahtar kelimelerle uygun analize yönlendirdim.\n\n"
                        + rep.text)
            return rep
        try:
            if self.kind == "ollama":
                return self._ask_ollama(question, history, max_steps)
            if self.kind == "openai":
                return self._ask_openai(question, history, max_steps)
            return self._ask_claude(question, history, max_steps)
        except Exception as e:          
            rep = fallback_answer(self.runner, question, reason="api_error")
            rep.text = explain_api_error(e, self.model, self.provider) + "\n\n" + rep.text
            rep.error = error_detail(e)
            return rep

    def _chat(self, messages):
        return _request(self.host, "/api/chat", self.api_key,
                        {"model": self.model, "messages": messages, "tools": OLLAMA_TOOLS, "stream": False})

    def _run_tool(self, name, args, calls):
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        out = self.runner.run(name, args or {})
        calls.append((name, args or {}, out))
        return out

    def _ask_ollama(self, question, history, max_steps) -> AdvisorReply:
        messages = [{"role": "system", "content": self.system_prompt}] + list(history or []) + \
                   [{"role": "user", "content": question}]
        calls = []
        for _ in range(max_steps):
            msg = self._chat(messages).get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return AdvisorReply(text=msg.get("content", ""), tool_calls=calls, mode="ollama")
            messages.append(msg)
            for tc in tool_calls:
                fn = tc.get("function") or {}
                out = self._run_tool(fn.get("name", ""), fn.get("arguments") or {}, calls)
                messages.append({"role": "tool", "tool_name": fn.get("name", ""), "content": _dump(out)})
        return AdvisorReply(text="Soru çok adımlı oldu; lütfen daha spesifik sorun.", tool_calls=calls, mode="ollama")

    def _ask_openai(self, question, history, max_steps) -> AdvisorReply:
        messages = [{"role": "system", "content": self.system_prompt}] + list(history or []) + \
                   [{"role": "user", "content": question}]
        calls = []
        for _ in range(max_steps):
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=OPENAI_TOOLS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return AdvisorReply(text=msg.content or "", tool_calls=calls, mode=self.provider)
            messages.append(_openai_assistant_turn(msg))
            for tc in msg.tool_calls:
                out = self._run_tool(tc.function.name, tc.function.arguments, calls)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _dump(out)})
        return AdvisorReply(text="Soru çok adımlı oldu; lütfen daha spesifik sorun.", tool_calls=calls, mode=self.provider)

    def _ask_claude(self, question, history, max_steps) -> AdvisorReply:
        messages = list(history or []) + [{"role": "user", "content": question}]
        calls = []
        for _ in range(max_steps):
            resp = self.client.messages.create(model=self.model, max_tokens=1500, system=self.system_prompt,
                                               tools=TOOLS, messages=messages)
            if resp.stop_reason != "tool_use":
                return AdvisorReply(text="".join(b.text for b in resp.content if b.type == "text"),
                                    tool_calls=calls, mode="claude")
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for blk in resp.content:
                if blk.type == "tool_use":
                    out = self._run_tool(blk.name, blk.input, calls)
                    results.append({"type": "tool_result", "tool_use_id": blk.id, "content": _dump(out)})
            messages.append({"role": "user", "content": results})
        return AdvisorReply(text="Soru çok adımlı oldu; lütfen daha spesifik sorun.", tool_calls=calls, mode="claude")


def ping_provider(provider: str, model: str, api_key: str = "", host: str | None = None) -> str:
    """Araçsız tek mesajlık bağlantı testi."""
    kind = PROVIDERS[provider]["kind"]
    msg = [{"role": "user", "content": "Sadece 'Hazırım' yaz."}]
    if kind == "ollama":
        return ping(host or ollama_settings()["host"], api_key, model)
    if kind == "openai":
        import openai
        c = openai.OpenAI(api_key=api_key, base_url=PROVIDERS[provider].get("base_url"))
        return c.chat.completions.create(model=model, messages=msg).choices[0].message.content or ""
    import anthropic
    r = anthropic.Anthropic(api_key=api_key).messages.create(model=model, max_tokens=20, messages=msg)
    return "".join(b.text for b in r.content if b.type == "text")


def error_detail(e: Exception) -> str:
    parts = [str(e)]
    body = getattr(e, "body", None)
    if body and not isinstance(body, str):
        parts.append(json.dumps(body, ensure_ascii=False, default=str))
    return " | ".join(p for p in parts if p)


def explain_api_error(e: Exception, model: str, provider: str = "ollama") -> str:
    """Dil modeli hatasını Türkçe ve çözüm önerisiyle açıklar."""
    msg = error_detail(e)
    code = getattr(e, "status_code", None)
    low = msg.lower()
    if provider == "ollama" and code is None and ("bağlantı" in low or "refused" in low or "timed out" in low):
        hint = "Ollama'ya bağlanılamadı. Ollama uygulamasının açık olduğundan emin olun."
    elif code == 404 or "not found" in low or "not_found" in low:
        hint = (f"`{model}` modeli bulunamadı. Kenar çubuğundaki Model alanını değiştirin"
                + (f" (yerelde: `ollama pull {model}`)." if provider == "ollama" else "."))
    elif code in (401, 403) or "api key" in low or "unauthorized" in low or "permission" in low:
        hint = "API anahtarı geçersiz veya bu modele erişim yetkisi yok."
    elif code in (402, 429) or "quota" in low or "credit" in low or "rate" in low:
        hint = "Kullanım/kota sınırı veya yetersiz kredi. Sağlayıcı hesabınızı kontrol edin."
    elif code == 400 and "tool" in low:
        hint = "Seçilen model araç çağırmayı desteklemiyor; araç destekli bir model seçin."
    else:
        hint = "Dil modeli isteği başarısız oldu."
    detail = msg[:600].replace("`", "'")
    return f"**Dil modeline ulaşılamadı.** {hint}\n\nHata ayrıntısı: `{detail}`\n\nBu arada soruyu yedek modla yanıtladım:"


# ---------------------------------------------------------------------------
_RULES = [
    (r"kaçan|sipariş verme|sessiz|gecik", "missing"),
    (r"birlikte|çapraz|cross|sepet|paket", "bundles"),
    (r"tahmin|gelecek|önümüzdeki|forecast", "demand_forecast"),
    (r"düş|azal|geril", "trend_down"),
    (r"artan|art[ıi]ş|yüksel|büyüyen|büyüme", "trend_up"),
    (r"risk|kayb|kaybed", "at_risk"),
    (r"segment|müşteri dna|müşteri grup", "segments"),
    (r"(ara|arama|potansiyel).*müşteri|müşteri.*(ara|potansiyel)|sıradaki 10", "calls"),
    (r"çok satan|az satan|düzenli|değişken|istikrarlı|dalgalı|portföy|dna", "portfolio"),
    (r"şehir|il bazında", "city"),
    (r"müşteri", "customers"),
    (r"kategori", "categories"),
    (r"ürün|kitap", "products"),
]


def fallback_answer(runner: ToolRunner, q: str, reason: str = "no_key") -> AdvisorReply:
    ql = q.lower()
    m = re.search(r"(?<![a-z0-9])(\d{1,3})(?![0-9])", ql)
    n = max(1, int(m.group(1))) if m else 10
    cust = re.search(r"\b([apsd]\d{5})\b", ql)
    if cust and re.search(r"öner|ürün", ql):
        name, args = "products_for_customer", {"customer_id": cust.group(1), "n": n}
    elif cust:
        name, args = "customer_profile", {"customer_id": cust.group(1)}
    else:
        intent = next((k for pat, k in _RULES if re.search(pat, ql)), "kpis")
        mapping = {
            "bundles": ("top_bundles", {"n": n}),
            "demand_forecast": ("demand_forecast", {"weeks": 4}),
            "trend_down": ("category_trend", {"months": 3, "direction": "down", "n": n}),
            "trend_up": ("category_trend", {"months": 3, "direction": "up", "n": n}),
            "at_risk": ("customer_segments", {"segment": "Riskli Müşteriler", "n": n}),
            "missing": ("missing_orders", {"n": n}),
            "segments": ("customer_segments", {}),
            "calls": ("top_call_list", {"n": n}),
            "portfolio": ("portfolio_products", {"dna": "Çok Satan – Değişken Talep" if ("değişken" in ql or "dalgalı" in ql) else "Çok Satan – Düzenli Talep", "n": n}),
            "city": ("top_entities", {"dimension": "city", "n": n}),
            "customers": ("top_entities", {"dimension": "customer", "n": n}),
            "categories": ("top_entities", {"dimension": "category", "n": n}),
            "products": ("top_entities", {"dimension": "product", "n": n}),
            "kpis": ("get_kpis", {}),
        }
        name, args = mapping[intent]
    out = runner.run(name, args)
    if reason == "no_key":
        text = f"**Yedek mod:** sorunuz `{name}` analizine yönlendirildi; sonuç aşağıdaki tabloda."
    else:
        text = f"Sorunuz `{name}` analizine yönlendirildi; sonuç aşağıdaki tabloda."
    if "error" in out:
        text += f"\n\n**Uyarı:** {out['error']}"
    elif name == "get_kpis":
        f = lambda x: f"{x:,.0f}".replace(",", ".")
        text += (f"\n\nToplam ciro **{f(out['revenue'])} TL**, {f(out['invoices'])} fatura, {f(out['customers'])} müşteri, "
                 f"{f(out['products'])} ürün. Ortalama iskonto %{100*out['avg_discount']:.1f}.".replace(".", ",", 0))
    return AdvisorReply(text=text, tool_calls=[(name, args, out)], mode="fallback")
