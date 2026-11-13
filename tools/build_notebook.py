import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "BookPulse_Analysis.ipynb"

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip()))


def code(s):
    CELLS.append(("code", s.strip()))


# ---------------------------------------------------------------------------
md("""
# BookPulse – Yapay Zekâ Destekli B2B Satış Zekâsı Platformu
""")

code("""
import sys, warnings
sys.path.insert(0, "..")        
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bookpulse import config, data, portfolio as pf, next_calls as nc, customer_dna as cd, basket as mb, forecast as fc
from bookpulse.ui import tr_df as tr      

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
plt.rcParams.update({"figure.figsize": (11, 4.5), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
""")
code("""
import importlib, platform
print("Python", platform.python_version())
for pkg in ["streamlit", "pandas", "numpy", "scipy", "sklearn", "plotly", "openpyxl", "matplotlib", "ollama"]:
    try:
        mod = importlib.import_module(pkg)
        print(f"  ✓ {pkg:<11} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"  ✗ {pkg:<11} kurulu değil -> pip install -r requirements.txt")

from bookpulse.advisor import ollama_status, setup_guide
durum = ollama_status()
print("\\nOllama durumu:", {"ready": "hazır", "no_model": "çalışıyor, model yok", "not_running": "kurulu, çalışmıyor",
                         "not_installed": "bulunamadı"}[durum["step"]],
      "| adres:", durum["host"], "| modeller:", durum["models"] or "-")
""")

# ---------------------------------------------------------------------------
md("""
## 0 · Veri hazırlığı ve temizliği
Ham Excel okunur ve `data.clean()` ile temizlenir. Fonksiyon, her çıkarılan satır için **gerekçe** döndürür.
""")
code("""
raw = data.load_raw()            
df, report = data.clean(raw)
df = data._add_fast_keys(df)
pd.Series(report, name="değer").to_frame()
""")
code("""
# Belge türleri: DPE/DPF de BAD/BSR ile aynı iskonto yapısında -> gerçek satış gibi davranıyor
raw_t = raw.assign(DocType=raw["InvoiceNo"].str[:3], Disc=1 - raw["NetPrice"] / raw["UnitPrice"])
raw_t.groupby("DocType").agg(**{"Satır": ("DocType", "size"), "Fatura": ("InvoiceNo", "nunique"),
                             "Müşteri": ("CustomerID", "nunique"), "Medyan İskonto": ("Disc", "median"),
                             "Medyan Adet": ("Quantity", "median")}).rename_axis("Belge türü")
""")
code("""
k = data.kpis(df)
print(f"Temiz satır: {len(df):,} | Ciro (Adet×NetPrice): {k['revenue']:,.0f} TL | Fatura: {k['invoices']:,} | "
      f"Müşteri: {k['customers']:,} | Ürün: {k['products']:,} | Ort. iskonto: %{100*k['avg_discount']:.1f}")

m = df.groupby("Month")["Revenue"].sum() / 1e6
ax = m.plot(kind="bar", color="#5B5FE3", title="Aylık ciro (milyon TL)")
ax.axvspan(2.5, 6.5, color="#FF4B4B", alpha=0.08, label="Covid kapanma dönemi")
ax.legend(); ax.set_xlabel(""); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 1 · Portföy DNA + Portföy Optimizasyonu
**DNA:** toplam adet (hacim) × aylık talep değişkenliği (CV). Eşikler, Yetersiz Satış Geçmişi dışındaki ürünlerin medyanı.
**Değişiklik:** CV artık ürünün ilk satış ayından itibaren hesaplanıyor. Takvim bazlı CV (`CV_Calendar`) karşılaştırma için tutuluyor.
""")
code("""
portfolio, th = pf.build_portfolio(df)
print("Eşikler:", {k: round(v, 3) if isinstance(v, float) else v for k, v in th.items()})
tr(pf.dna_summary(portfolio))
""")
code("""
# Yaşam döngüsü CV'si hangi ürünlerin sınıfını değiştirdi? (takvim CV'si ile sınıflansaydı)
core = portfolio[portfolio["DNA"] != "Yetersiz Satış Geçmişi"]
old_volatile = core["CV_Calendar"] > th["cv_threshold"]
new_volatile = core["CV"] > th["cv_threshold"]
print(f"Takvim CV'sine göre 'değişken' görünüp aslında istikrarlı olan ürün: {(old_volatile & ~new_volatile).sum():,}")
print("Bu ürünlerin ilk satış ayları:")
core[old_volatile & ~new_volatile]["First_Month"].value_counts().sort_index().tail(8).rename_axis("İlk satış ayı").rename("Ürün sayısı").to_frame()
""")
code("""
fig, ax = plt.subplots(figsize=(11, 6))
for dna in ["Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep", "Az Satan – Düzenli Talep", "Az Satan – Değişken Talep"]:
    d = core[core["DNA"] == dna]
    ax.scatter(d["Total_Qty"], d["CV"], s=4, alpha=0.35, label=f"{dna} ({len(d):,})", color=pf.DNA_COLORS[dna])
ax.axvline(th["volume_threshold"], ls="--", c="gray"); ax.axhline(th["cv_threshold"], ls="--", c="gray")
ax.set_xscale("log"); ax.set_xlabel("Toplam adet (log)"); ax.set_ylabel("CV (ilk satıştan itibaren)")
ax.set_title("Portföy DNA haritası"); ax.legend(markerscale=4); plt.tight_layout(); plt.show()
""")
md("""
### Portföy Optimizasyonu – aksiyon önerileri
Her ürüne **performans puanı** (ciro %35, müşteri erişimi %20, trend %15, istikrar %15, düşük iskonto %15 – yüzdelik sıralar) ve
DNA + trend + yenilik kurallarıyla bir **aksiyon** atanır.
""")
code("""
tr(pf.action_summary(portfolio))
""")
code("""
cols = ["Description", "Category", "DNA", "Performance_Score", "Revenue", "Total_Qty", "Trend", "Customers"]
for act in ["Sürekli Bulunurluğu Değerlendir", "İzle (Düşüşte)", "Tanıtım Fırsatı", "Ürün Performansını İncele"]:
    print(f"\\n=== {act}: {pf.ACTIONS[act]}")
    print(portfolio[portfolio["Action"] == act].sort_values("Revenue", ascending=False)[cols].head(5).pipe(tr).to_string())
""")
code("""
# Kategori bazında portföy yoğunlaşması
tr(pf.category_concentration(portfolio, top=12))
""")

# ---------------------------------------------------------------------------
md("""
## 2 · Sıradaki 10 Arama v2
v1'e eklenenler: **kategori payı**, **sipariş sıklığı** (sipariş/aktif ay), **ürün çeşitliliği**, **benzer ürün sinyali**
(ürünle aynı müşterilerce alınan ürünler – ürün tabanlı işbirlikçi filtreleme), **uyuyan müşteri** işareti ve her öneri için **gerekçe**.
Ters yön de var: *müşteriye hangi ürünler önerilmeli?*
""")
code("""
engine = nc.NextCallsEngine(df, portfolio)
top_star = portfolio[portfolio["DNA"] == "Çok Satan – Düzenli Talep"]["Revenue"].idxmax()
print("Ürün:", portfolio.loc[top_star, "Description"], "|", portfolio.loc[top_star, "Category"])
rec = engine.recommend(top_star, n=10)
tr(rec.drop(columns=["StockCode", "Product", "Reason"]))
""")
code("""
print("Örnek gerekçe →", rec.loc[0, "Reason"])
""")
code("""
# Ekibin ilk modeli (v1: kategori miktarı %50, sipariş %30, yakınlık %20) aynı motorla da çalışır
v1 = engine.recommend(top_star, n=10, weights=nc.V1_WEIGHTS)
tr(v1[["CustomerID", "Score", "Category_Qty", "Category_Orders", "Days_Since_Last"]])
""")
md("""
### Çift yönlü geçiş: Portföy DNA → Sıradaki 10 Arama → Müşterinin Portföy DNA'sı → yeni ürün
Listeden bir müşteri seçildiğinde, o müşterinin **tüm alımlarının** ürün DNA gruplarına dağılımı şirket geneliyle karşılaştırılır.
Ardından müşterinin henüz almadığı ürünlerden biri seçilip yeniden Sıradaki 10 Arama listesi oluşturulabilir (panoda tek tıkla).
""")
code("""
musteri = rec.loc[0, "CustomerID"]
cp = engine.customer_portfolio(musteri)
ax = cp[["Customer_Share_%", "Company_Share_%"]].rename(
    columns={"Customer_Share_%": musteri, "Company_Share_%": "Şirket geneli"}).plot(
    kind="bar", color=["#FF4B4B", "#C7DEFC"], title=f"{musteri}: alımların Portföy DNA dağılımı (adet %)")
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
tr(cp)
""")
code("""
# 'Neden bu müşteri?' kanıtları
neden = engine.why_customer(musteri, top_star)
print(f"Kategori payı: %{neden['category_share_%']} | ürünle birlikte alınan ürünlerden {neden['n_similar_products_bought']} tanesini almış")
print("Örnekler:", ", ".join(neden["similar_products_bought"][:5]))
# Döngü: müşteriye önerilen ilk ürün için yeni arama listesi
yeni_urun = engine.products_for_customer(musteri, n=1, dna_filter=None)["StockCode"].iloc[0]
print("\\nYeni ürün:", engine.products.loc[yeni_urun, "Description"])
tr(engine.recommend(yeni_urun, n=5)[["CustomerID", "Score", "Priority", "Reason"]])
""")
code("""
# Ters yön: en büyük müşteriye önerilecek ürünler
cust = df.groupby("CustomerID")["Revenue"].sum().idxmax()
print("Müşteri:", cust)
tr(engine.products_for_customer(cust, n=8))
""")
md("""
### Geriye dönük test
Ağırlıklar **Ocak–Haziran → Temmuz–Eylül** doğrulama penceresinde seçildi. Nihai başarı ayrı bir pencerede ölçülür:
model **Oca–Eyl** verisiyle kurulur, 300 rastgele Çok Satan – Düzenli Talep / Çok Satan – Değişken Talep ürünü için önerilen 10 müşterinin
ürünü **Eki–Ara**'da gerçekten alıp almadığına bakılır.
""")
code("""
bt = nc.backtest(df, n_products=300)
tr(bt["table"])
""")
code("""
ax = bt["table"]["Precision@10_%"].plot(kind="barh", color=["#5B5FE3", "#FF8A8A", "#C9CEDC"],
                                        title="İsabet@10 – önerilen 10 müşteriden kaçı ürünü gerçekten aldı (%)")
ax.invert_yaxis(); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 3 · Müşteri DNA – RFM + K-Means
RFM skorları (1-5) ile iş dilinde segmentler; ayrıca log-ölçekli R/F/M + kategori çeşitliliği üzerinde K-Means (k silhouette ile).
""")
code("""
rfm = cd.rfm_table(df)
customers, info = cd.kmeans_segments(rfm)
tr(cd.segment_summary(customers))
""")
code("""
print("Silhouette:", {k: round(v, 3) for k, v in info["silhouette"].items()}, "→ k =", info["best_k"])
tr(info["profile"])
""")
code("""
s = tr(cd.segment_summary(customers)[["Customer_Share_%", "Revenue_Share_%"]])
s.plot(kind="bar", title="Segment: müşteri payı vs ciro payı (%)", color=["#C7DEFC", "#5B5FE3"])
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()
""")
code("""
# Müşteri DNA × Portföy DNA: hangi segment hangi ürün grubundan alıyor? (satır %)
x = cd.segment_x_portfolio(df, customers, portfolio)
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(x.values, cmap="Blues")
ax.set_xticks(range(len(x.columns)), x.columns, rotation=20); ax.set_yticks(range(len(x.index)), x.index)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        ax.text(j, i, f"{x.values[i, j]:.0f}", ha="center", va="center", fontsize=9)
ax.set_title("Segment cirosunun Portföy DNA dağılımı (%)"); ax.grid(False); plt.tight_layout(); plt.show()
""")
md("""
### Kaçan Sipariş Dedektörü
Her müşterinin siparişleri arasındaki tipik süre (medyan gün) hesaplanır; son siparişten bu yana geçen süre bunun en az
2 katıysa (ve en az 21 gün) müşteri işaretlenir. Geriye dönük kontrol: 30 Eylül itibarıyla işaretlenenlerin kaçı
Ekim–Aralık'ta yeniden sipariş verdi?
""")
code("""
from bookpulse.ui import pct
mo = cd.missing_orders(df)
print(f"İşaretlenen müşteri: {len(mo)} · yıllık cirosu {mo['Revenue'].sum():,.0f} TL "
      f"(toplamın {pct(100 * mo['Revenue'].sum() / df['Revenue'].sum(), 1)})")
egitim, test_ = df[df["InvoiceDate"] <= "2020-09-30"], df[df["InvoiceDate"] > "2020-09-30"]
mo_eylul = cd.missing_orders(egitim)
geri = mo_eylul.index.isin(test_["CustomerID"].unique())
print(f"30 Eylül'de işaretlenen {len(mo_eylul)} müşterinin {pct(100 * geri.mean())} Ekim–Aralık'ta geri döndü, "
      f"{pct(100 * (1 - geri.mean()))} dönmedi.")
tr(mo.head(10))
""")
md("""
> **Bulgu:** *Riskli Müşteriler* müşterilerin cirosunun yarıdan fazlası **Çok Satan – Değişken Talep** ürünlerinden geliyor; yani bunlar
> büyük olasılıkla sezonluk (ör. okul dönemi) alıcılar. Geri kazanım aramaları sezon öncesine planlanmalı.
""")

# ---------------------------------------------------------------------------
md("""
## 4 · Sepet Analizi
Sepet = fatura (2–150 kalem; çok büyük B2B faturaları sahte birliktelik ürettiği için dışarıda).
Apriori'nin 2'li kural adımı seyrek matris çarpımıyla hesaplanır. Metrikler: destek (support), güven (confidence), kaldıraç (lift).
""")
code("""
rules = mb.association_rules(df, level="product")
print(f"{rules.attrs['n_baskets']:,} sepet, {len(rules):,} kural")
tr(rules[["A_Name", "B_Name", "Count_AB", "Support", "Confidence", "Lift"]].head(8))
""")
code("""
# Farklı kategorilerden paket önerileri (aynı serinin ciltleri yerine)
tr(mb.bundles(rules, n=10, min_confidence=0.15, cross_category_only=True)[["A_Name", "A_Category", "B_Name", "B_Category", "Confidence", "Lift"]])
""")
code("""
# Çapraz satış: en çok satan ürünü alanlara ne önerilmeli?
tr(mb.cross_sell(rules, top_star, n=8)[["B_Name", "B_Category", "Count_AB", "Confidence", "Lift"]])
""")
code("""
cat_rules = mb.association_rules(df, level="category", min_support_count=50, min_lift=1.5)
tr(cat_rules[["A", "B", "Count_AB", "Confidence", "Lift"]].head(10))
""")

# ---------------------------------------------------------------------------
md("""
## 5 · Talep Tahmini
Veri yalnızca 2020'yi (≈51 hafta) kapsıyor ve Covid kırılmaları var → yıllık mevsimsellik öğrenilemez.
Bu yüzden ilk 20 kategori + toplam için **haftalık** seriler kurulur, dört yöntem son 8 haftada karşılaştırılır
ve **her seri için doğrulamada en iyi yöntem** seçilir.
""")
code("""
S = fc.weekly_series(df, level="Category", top_n=20)
ev = fc.evaluate(S, test_weeks=8)
print("Kesim:", ev["cutoff"], "| GBM:", ev["gbm_name"])
ev["overall"].rename_axis("Yöntem")
""")
code("""
ev["per_series"]["Best"].value_counts().rename_axis("En iyi yöntem").rename("Seri sayısı").to_frame()
""")
md("""
> **Dürüst sonuç:** Bu veri setinde gradient boosting, basit Naif/SES yöntemlerini **geçemiyor** (WAPE ≈ %25 vs %15).
> 2020'deki rejim değişiklikleri (kapanma → toparlanma → okul sezonu) karmaşık modelin öğrendiği örüntülerin
> test döneminde geçerli olmamasına yol açıyor. Daha uzun geçmiş (2019, 2021) eklendiğinde XGBoost/Prophet yeniden denenmeli.
> Ürün bazında haftalık talep çok gürültülü (WAPE > %70); ürün düzeyinde tahmin yerine **satış hızı** göstergesi önerilir.
""")
code("""
F = fc.forecast(S, horizon=8, evaluation=ev)
s = "TOPLAM"
f = F[F["Series"] == s]
fig, ax = plt.subplots()
ax.plot(S.index, S[s], marker="o", ms=3, color="#5B5FE3", label="Gerçekleşen")
best = ev["per_series"].loc[s, "Best"]
ax.plot(ev["preds"][best].index, ev["preds"][best][s], ls=":", color="#7CB8F7", label=f"Doğrulama ({best})")
ax.plot(f["Week"], f["Forecast"], lw=2.5, color="#FF4B4B", label="Tahmin")
ax.fill_between(f["Week"], f["Lower"], f["Upper"], color="#FF4B4B", alpha=0.12, label="%80 tahmin aralığı")
ax.set_title("Toplam haftalık adet ve 8 haftalık tahmin"); ax.legend(); plt.tight_layout(); plt.show()
""")
code("""
tr(fc.outlook(S, F, weeks=4).head(10))
""")

# ---------------------------------------------------------------------------
md("""
## 6 · Yapay Zekâ Satış Danışmanı (Ollama)
Danışman, dil modelini bilgisayarınızda çalıştıran ücretsiz **Ollama** ile çalışır. Model ham veriye serbestçe erişmez;
yalnızca aşağıdaki **kontrollü analiz fonksiyonlarını** (araç) çağırabilir, bu yüzden cevaplardaki sayılar gerçek analiz
sonuçlarından gelir. Ollama bulunamazsa kurulum adımları gösterilir.
""")
code("""
from bookpulse.advisor import TOOLS, SalesAdvisor, ping
from bookpulse.engine import BookPulse
from IPython.display import Markdown, display

pd.DataFrame([(t["name"], t["description"]) for t in TOOLS], columns=["araç", "açıklama"])
""")
code("""
# 6.1 Ollama kontrolü: bulunamazsa kurulum rehberi gösterilir
durum = ollama_status()
if durum["step"] == "ready":
    print(f"Ollama hazır · model: {durum['model']} · kurulu modeller: {durum['models']}")
    print("Bağlantı testi (araçsız tek mesaj):", ping(durum["host"], "", durum["model"]))
else:
    display(Markdown(setup_guide(durum)))
""")
code("""
# 6.2 Danışmana soru sor (Ollama varsa doğal dilde cevap + kullanılan araçlar; yoksa anahtar kelime modu)
bp = BookPulse(df)
advisor = SalesAdvisor(bp, model=durum["model"], available=durum["step"] == "ready")

def sor(soru):
    reply = advisor.ask(soru)
    display(Markdown(f"**Soru:** {soru}\\n\\n**Cevap ({advisor.mode}):** {reply.text}"))
    for name, args, out in reply.tool_calls:
        print(f"  araç: {name} {args}")
        if "table" in out:
            t = out["table"]
            display(tr(pd.DataFrame(t["rows"], columns=t["columns"]).head(8)))
    return reply

_ = sor("Son üç ayda satışları düşen kategoriler hangileri?")
""")
code("""
_ = sor("Şeker Portakalı için hangi müşterileri aramalıyım?")
""")
code("""
_ = sor("Kaybetme riski olan değerli müşteriler kimler?")
""")


# ---------------------------------------------------------------------------
def execute(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.chdir(ROOT / "notebooks")
    ns = {"__name__": "__main__"}
    current = {"outs": None}
    try:
        import IPython 
    except ImportError:                      i
        import types
        ip = types.ModuleType("IPython")
        ipd = types.ModuleType("IPython.display")

        class Markdown:
            def __init__(self, data):
                self.data = data

        def display(obj):
            if isinstance(obj, Markdown):
                data = {"text/markdown": obj.data, "text/plain": obj.data}
            else:
                data = rich(obj)
            current["outs"].append({"output_type": "display_data", "metadata": {}, "data": data})
        ipd.Markdown, ipd.display = Markdown, display
        ip.display, ip.version_info, ip.get_ipython = ipd, (8, 24, 0), (lambda: None)
        sys.modules["IPython"], sys.modules["IPython.display"] = ip, ipd
    outputs_all = []
    count = 0

    def rich(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            o = obj.to_frame() if isinstance(obj, pd.Series) else obj
            return {"text/html": o.to_html(max_rows=40), "text/plain": o.to_string(max_rows=40)}
        return {"text/plain": repr(obj)}

    for kind, src in cells:
        if kind != "code":
            outputs_all.append(None)
            continue
        count += 1
        outs = []
        current["outs"] = outs
        buf = io.StringIO()
        try:
            tree = ast.parse(src)
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            body = tree.body[:-1] if last else tree.body
            with contextlib.redirect_stdout(buf):
                exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), ns)
                val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns) if last else None
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            for num in plt.get_fignums():
                b = io.BytesIO()
                plt.figure(num).savefig(b, format="png", dpi=90, bbox_inches="tight")
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
            plt.close("all")
            if val is not None and not (hasattr(val, "__module__") and "matplotlib" in str(type(val))):
                outs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": rich(val)})
        except Exception:
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            outs.append({"output_type": "error", "ename": "Error", "evalue": tb.splitlines()[-1], "traceback": tb.splitlines()})
        outputs_all.append((count, outs))
        print(f"cell {count} ok", file=sys.stderr)
    return outputs_all


def build(execute_cells=True):
    results = execute(CELLS) if execute_cells else [None] * len(CELLS)
    nb_cells = []
    for (kind, src), res in zip(CELLS, results):
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cnt, outs = res if res else (None, [])
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": cnt, "outputs": outs, "source": lines})
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("yazıldı:", OUT)


if __name__ == "__main__":
    build(execute_cells="--no-exec" not in sys.argv)
