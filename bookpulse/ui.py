"""BookPulse görsel kimliği: renk paleti, global CSS, Plotly teması ve HTML bileşenleri.

Renklerin anlamı bütün uygulamada aynıdır:
  * Indigo / mavi   -> normal bilgi, ana veri serisi, seçim, olumlu durum
  * Amber           -> dikkat, dalgalı/düzensiz davranış
  * Kırmızı         -> risk, düşüş, acil aksiyon (yalnızca bu anlamda kullanılır)
  * Gri             -> nötr / yetersiz veri
"""
from __future__ import annotations

import html

# ---------------------------------------------------------------------------
PALETTE = {
    "red": "#FF4B4B", "red_600": "#E03A3A", "coral": "#FF8A8A", "rose": "#FFC9C9", "red_50": "#FFF1F1",
    "indigo": "#5B5FE3", "indigo_700": "#3F43C4", "blue": "#3B82F6", "sky": "#7CB8F7", "ice": "#C7DEFC",
    "indigo_50": "#EEF0FF", "navy": "#14163A",
    "ink": "#1E2235", "text": "#3A3F55", "muted": "#7B8199", "line": "#E8EAF2",
    "bg": "#F6F7FB", "card": "#FFFFFF", "green": "#16A34A",
    "amber": "#F59E0B", "amber_50": "#FFF7E6", "gray": "#C9CEDC",
}
P = PALETTE

# Kategorik seriler: önce mavi aile (kırmızı risk için ayrılmıştır)
COLORWAY = [P["indigo"], P["sky"], P["amber"], P["indigo_700"], P["ice"], P["blue"], "#A5A8F0", P["gray"]]
SEQ_BLUE = [[0, "#F3F5FF"], [0.35, P["ice"]], [0.7, P["sky"]], [1, P["indigo"]]]
SEQ_RED = [[0, "#FFF5F5"], [0.35, P["rose"]], [0.7, P["coral"]], [1, P["red"]]]
DIVERGING = [[0, P["red"]], [0.5, "#F5F6FA"], [1, P["indigo"]]]

DNA_COLORS = {
    "Çok Satan – Düzenli Talep": P["indigo"], "Çok Satan – Değişken Talep": P["amber"], "Az Satan – Düzenli Talep": P["sky"],
    "Az Satan – Değişken Talep": P["coral"], "Yetersiz Satış Geçmişi": P["gray"],
}
SEG_COLORS = {
    "En Değerli Müşteriler": P["indigo"], "Sadık Müşteriler": P["blue"], "Gelişen Müşteriler": P["sky"],
    "Yeni Gözlenen Müşteriler": P["ice"], "İlgi Gerektiren Müşteriler": P["amber"], "Riskli Müşteriler": P["coral"],
    "Riskli Değerli Müşteriler": P["red"], "Pasif Müşteriler": P["gray"],
}

FONT = "'Source Sans 3', 'Source Sans Pro', -apple-system, 'Segoe UI', sans-serif"


# ---------------------------------------------------------------------------
def plotly_template():
    """'bookpulse' Plotly temasını kaydeder ve varsayılan yapar."""
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=P["text"]),
        title=dict(font=dict(size=17, color=P["ink"], family=FONT), x=0, xanchor="left", y=0.97, pad=dict(l=4)),
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=56, b=12),
        separators=",.",                         
        xaxis=dict(showgrid=False, linecolor=P["line"], tickcolor=P["line"], ticks="outside", ticklen=4,
                   zeroline=False, automargin=True, title=dict(font=dict(color=P["muted"], size=12))),
        yaxis=dict(gridcolor="#EEF0F6", gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)", automargin=True,
                   title=dict(font=dict(color=P["muted"], size=12))),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color=P["muted"]), title=dict(text="")),
        hoverlabel=dict(bgcolor=P["navy"], bordercolor=P["navy"], font=dict(color="white", family=FONT, size=12)),
        coloraxis=dict(colorscale=SEQ_BLUE, colorbar=dict(outlinewidth=0, thickness=10)),
        bargap=0.28,
    )
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0), cornerradius=6))]
    t.data.scatter = [go.Scatter(line=dict(width=2.6))]
    pio.templates["bookpulse"] = t
    pio.templates.default = "bookpulse"
    px.defaults.template = "bookpulse"
    px.defaults.color_discrete_sequence = COLORWAY
    px.defaults.color_continuous_scale = SEQ_BLUE


# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {{ font-family: {FONT}; }}
.stApp {{ background: {P['bg']}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ color: {P['ink']}; letter-spacing: -0.02em; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {P['line']}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    padding: 9px 12px; border-radius: 10px; margin: 0; width: 100%;
    transition: background .15s ease; cursor: pointer;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{ display: none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{ font-size: 15px; color: {P['text']}; font-weight: 500; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background: {P['bg']}; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{ background: {P['indigo_50']}; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{ color: {P['indigo_700']}; font-weight: 700; }}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: {P['muted']}; font-weight: 700;
}}

/* ---------- kartlar: grafik, tablo, metrik, kenarlıklı kapsayıcı ---------- */
[data-testid="stPlotlyChart"], [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    background: {P['card']}; border-radius: 16px;
}}
[data-testid="stPlotlyChart"] {{
    padding: 10px 14px 6px; border: 1px solid {P['line']};
    box-shadow: 0 1px 2px rgba(20,22,58,.04), 0 8px 24px -12px rgba(20,22,58,.12);
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {P['line']}; border-radius: 14px; overflow: hidden; background: #fff;
    box-shadow: 0 8px 24px -16px rgba(20,22,58,.18);
}}
[data-testid="stMetric"] {{
    padding: 16px 18px; border: 1px solid {P['line']}; box-shadow: 0 8px 24px -16px rgba(20,22,58,.18);
}}
[data-testid="stMetricLabel"] p {{ color: {P['muted']}; font-size: 13px; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {P['ink']}; font-weight: 700; font-size: 1.55rem; }}
[data-testid="stExpander"] details {{ background: #fff; border: 1px solid {P['line']}; border-radius: 14px; }}

/* ---------- sekmeler ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {P['line']}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 6px; font-weight: 600; color: {P['muted']}; }}
.stTabs [data-baseweb="tab"]:hover {{ color: {P['red']}; }}
.stTabs [aria-selected="true"] {{ color: {P['red']} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {P['red']}; height: 3px; border-radius: 3px; }}

/* ---------- butonlar ---------- */
.stButton > button, .stDownloadButton > button {{
    border-radius: 10px; font-weight: 600; border: 1px solid {P['line']}; background: #fff; color: {P['ink']};
    padding: .45rem 1rem; transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {P['red']}; color: {P['red']}; background: {P['red_50']}; }}
.stButton > button[kind="primary"] {{ background: {P['red']}; color: #fff; border-color: {P['red']}; }}
.stButton > button[kind="primary"]:hover {{ background: {P['red_600']}; color: #fff; }}

/* ---------- uyarı kutuları ---------- */
[data-testid="stAlert"] {{ border-radius: 12px; border: none; }}

/* ---------- sohbet ---------- */
[data-testid="stChatMessage"] {{ background: #fff; border: 1px solid {P['line']}; border-radius: 16px; padding: 14px 16px; }}
[data-testid="stChatInput"] textarea {{ font-size: 15px; }}

/* ---------- özel bileşenler ---------- */
.bp-brand {{ display:flex; align-items:center; gap:10px; padding: 4px 4px 14px; }}
.bp-brand .logo {{
    width: 38px; height: 38px; border-radius: 11px; display:grid; place-items:center;
    background: linear-gradient(135deg, {P['red']} 0%, {P['indigo']} 100%); color:#fff; font-weight:800; font-size:18px;
    box-shadow: 0 8px 18px -8px rgba(91,95,227,.7);
}}
.bp-brand .name {{ font-size: 20px; font-weight: 800; color: {P['ink']}; letter-spacing:-.02em; line-height:1; }}
.bp-brand .tag {{ font-size: 11.5px; color: {P['muted']}; margin-top:3px; }}

.bp-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin: 4px 0 18px; }}
.bp-header .eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing:.1em; text-transform: uppercase; color: {P['red']}; }}
.bp-header h1 {{ font-size: 2.05rem; font-weight: 800; margin: 2px 0 4px; padding:0; color:{P['ink']}; }}
.bp-header p {{ color: {P['muted']}; margin: 0; font-size: 15px; max-width: 820px; }}
.bp-chip {{
    display:inline-flex; align-items:center; gap:6px; padding: 6px 12px; border-radius: 999px; font-size: 12.5px;
    font-weight: 600; background: #fff; border: 1px solid {P['line']}; color: {P['text']}; white-space: nowrap;
}}
.bp-chip .dot {{ width:8px; height:8px; border-radius:50%; background:{P['green']}; }}

.bp-hero {{
    position:relative; border-radius: 20px; padding: 26px 28px; color: #fff; margin-bottom: 16px;
    background: radial-gradient(1200px 300px at 85% -20%, rgba(255,75,75,.55), transparent 60%),
                radial-gradient(900px 300px at 10% 120%, rgba(91,95,227,.75), transparent 60%),
                {P['navy']};
    box-shadow: 0 20px 40px -24px rgba(20,22,58,.6);
}}
.bp-hero .label {{ font-size: 13px; opacity:.75; font-weight:600; letter-spacing:.04em; }}
.bp-hero .value {{ font-size: 2.6rem; font-weight: 800; letter-spacing:-.03em; line-height:1.1; margin-top:4px; }}
.bp-hero .sub {{ font-size: 14px; opacity: .8; margin-top: 6px; }}
.bp-hero .grid {{ display:grid; grid-template-columns: 1.4fr repeat(3, 1fr); gap: 18px; align-items:end; }}
.bp-hero .mini {{ background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12); border-radius: 14px; padding: 12px 14px; }}
.bp-hero .mini .v {{ font-size: 1.35rem; font-weight: 700; }}
.bp-hero .mini .l {{ font-size: 12px; opacity: .7; }}

.bp-kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 4px 0 18px; }}
.bp-kpi {{
    background:#fff; border:1px solid {P['line']}; border-radius: 16px; padding: 16px 18px;
    box-shadow: inset 4px 0 0 var(--accent), 0 8px 24px -16px rgba(20,22,58,.18); position: relative;
}}
.bp-kpi .top {{ display:flex; align-items:center; justify-content:space-between; }}
.bp-kpi .label {{ color:{P['muted']}; font-size:13px; font-weight:600; }}
.bp-kpi .icon {{ width:30px; height:30px; border-radius:9px; display:grid; place-items:center; font-size:15px;
                 background: var(--soft); color: var(--accent); }}
.bp-kpi .value {{ color:{P['ink']}; font-size: clamp(1.15rem, 1.6vw, 1.6rem); font-weight: 800; letter-spacing:-.02em;
  margin-top: 6px; line-height:1.15; overflow-wrap:anywhere; }}
.bp-kpi .note {{ font-size: 12.5px; color:{P['muted']}; margin-top: 2px; }}
.bp-kpi .note b.up {{ color: {P['indigo']}; }} .bp-kpi .note b.down {{ color: {P['red']}; }}

.bp-section {{ display:flex; align-items:center; gap:10px; margin: 22px 0 10px; }}
.bp-section h3 {{ font-size: 1.1rem; margin:0; padding:0; }}
.bp-section .bar {{ width: 4px; height: 18px; border-radius: 4px; background: linear-gradient({P['red']}, {P['indigo']}); }}
.bp-section span {{ color:{P['muted']}; font-size: 13px; }}

.bp-callout {{ border-radius: 14px; padding: 14px 16px; margin: 8px 0 14px; font-size: 14.5px;
               background: {P['indigo_50']}; color: {P['indigo_700']}; border: 1px solid #DCDFFF; }}
.bp-callout.red {{ background: {P['red_50']}; color: {P['red_600']}; border-color: #FFD9D9; }}
.bp-callout b {{ font-weight: 700; }}

@media (max-width: 900px) {{ .bp-hero .grid {{ grid-template-columns: 1fr 1fr; }} }}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CSS + TOOLTIP_CSS + NAV_CSS, unsafe_allow_html=True)


def setup():
    inject_css()
    plotly_template()
    localize_plotly_express()


# ---------------------------------------------------------------------------
def _e(x) -> str:
    return html.escape(str(x))


def brand() -> str:
    return ('<div class="bp-brand"><div class="logo">B</div><div><div class="name">BookPulse</div>'
            '<div class="tag">Doğru Ürün · Doğru Müşteri · Doğru Zaman</div></div></div>')


def page_header(title: str, subtitle: str = "", eyebrow: str = "", chip: str | None = None) -> str:
    chip_html = f'<span class="bp-chip"><span class="dot"></span>{_e(chip)}</span>' if chip else ""
    return (f'<div class="bp-header"><div>'
            f'{f"<div class=eyebrow>{_e(eyebrow)}</div>" if eyebrow else ""}'
            f'<h1>{_e(title)}</h1>{f"<p>{glossify(_e(subtitle))}</p>" if subtitle else ""}</div>{chip_html}</div>')


def section(title: str, note: str = "") -> str:
    return f'<div class="bp-section"><div class="bar"></div><h3>{_e(title)}</h3><span>{glossify(_e(note))}</span></div>'


def kpi_cards(items: list[dict]) -> str:
    """items: {label, value, note?, icon?, tone: 'indigo'|'red'|'blue'|'coral'}"""
    tones = {"indigo": (P["indigo"], P["indigo_50"]), "red": (P["red"], P["red_50"]),
             "blue": (P["blue"], "#EAF2FE"), "coral": (P["coral"], P["red_50"])}
    cards = []
    for it in items:
        acc, soft = tones.get(it.get("tone", "indigo"), tones["indigo"])
        note = it.get("note", "")
        cards.append(
            f'<div class="bp-kpi" style="--accent:{acc};--soft:{soft}"><div class="top">'
            f'<span class="label">{glossify(_e(it["label"]))}</span><span class="icon">{icon(it["icon"]) if it.get("icon") else ""}</span></div>'
            f'<div class="value">{_e(it["value"])}</div><div class="note">{glossify(note)}</div></div>')
    return f'<div class="bp-kpis">{"".join(cards)}</div>'


def hero(label: str, value: str, sub: str, minis: list[tuple[str, str]]) -> str:
    m = "".join(f'<div class="mini"><div class="v">{_e(v)}</div><div class="l">{glossify(_e(l))}</div></div>' for l, v in minis)
    return (f'<div class="bp-hero"><div class="grid"><div><div class="label">{glossify(_e(label))}</div>'
            f'<div class="value">{_e(value)}</div><div class="sub">{glossify(_e(sub))}</div></div>{m}</div></div>')


def callout(text_html: str, tone: str = "indigo") -> str:
    return f'<div class="bp-callout {"red" if tone == "red" else ""}">{glossify(text_html)}</div>'


# ---------------------------------------------------------------------------
COLS = {
    # genel
    "CustomerID": "Müşteri", "StockCode": "Ürün Kodu", "Description": "Ürün", "Product": "Ürün", "Category": "Kategori",
    "City": "Şehir", "Revenue": "Ciro (₺)", "Quantity": "Adet", "Orders": "Sipariş", "Customers": "Müşteri Sayısı",
    "Products": "Ürün Sayısı", "Month": "Ay", "Share_%": "Pay %", "Avg_Discount": "Ort. İskonto",
    # portföy
    "Total_Qty": "Toplam Adet", "Window_Months": "Satış Penceresi (ay)", "Monthly_Avg": "Aylık Ort. Adet",
    "Monthly_Std": "Aylık Std. Sapma", "CV": "Değişkenlik (CV)", "CV_Calendar": "CV (takvim)",
    "Active_Months": "Aktif Ay", "First_Month": "İlk Satış Ayı", "Last_Month": "Son Satış Ayı", "Peak_Month": "Zirve Ay",
    "Qty_Last3": "Son 3 Ay Adet", "Qty_Prev3": "Önceki 3 Ay Adet", "Last_Sale": "Son Satış", "Trend": "Trend",
    "Days_Since_Sale": "Satışsız Gün", "Performance_Score": "Performans Puanı", "Action": "Aksiyon",
    "Action_Reason": "Aksiyon Gerekçesi", "Product_Share_%": "Ürün Payı %", "Qty_Share_%": "Adet Payı %",
    "Revenue_Share_%": "Ciro Payı %", "Stable_Stars": "Çok Satan – Düzenli Ürün Sayısı", "Limited_History_%": "Yetersiz Satış Geçmişi %",
    "HHI": "HHI (yoğunlaşma)", "Top10%_Product_Rev_Share": "İlk %10 Ürünün Ciro Payı",
    # sıradaki 10 arama
    "Score": "Puan", "Category_Qty": "Kategori Adedi", "Category_Orders": "Kategori Siparişi",
    "Distinct_Products": "Farklı Ürün", "Similar_Score": "Benzerlik Puanı", "Days_Since_Last": "Son Alımdan Beri (gün)",
    "Dormant": "Uyuyan", "Priority": "Öncelik", "Reason": "Gerekçe", "Neighbors_Bought": "Alan Benzer Müşteri",
    "Category_Affinity": "Kategori Yakınlığı", "CF_Score": "Benzer Müşteri Puanı",
    "Precision@10_%": "İsabet@10 %", "Hit_Rate_%": "En Az 1 İsabet %", "Hits": "İsabet Sayısı",
    # müşteri
    "Recency": "Son Alım (gün)", "Frequency": "Sipariş Sayısı", "Monetary": "Ciro (₺)", "Tenure_Days": "Müşteri Yaşı (gün)",
    "Categories": "Kategori Sayısı", "CustomerType": "Müşteri Tipi", "Avg_Order_Value": "Ort. Sipariş Tutarı",
    "RFM_Score": "RFM Skoru", "Cluster": "Küme No", "Cluster_Name": "Küme", "Recency_med": "Medyan Son Alım (gün)",
    "Frequency_med": "Medyan Sipariş", "AOV_med": "Medyan Sipariş Tutarı", "Customer_Share_%": "Müşteri Payı %",
    "silhouette": "Silhouette", "k": "Küme sayısı (k)",
    # sepet
    "A_Name": "Ürün A", "B_Name": "Ürün B", "A_Category": "Kategori A", "B_Category": "Kategori B",
    "A": "A", "B": "B", "Count_AB": "Birlikte Görülen Sepet", "Support": "Destek", "Confidence": "Güven",
    "Lift": "Kaldıraç (lift)", "Support_B": "Destek (B)",
    # tahmin
    "Week": "Hafta", "Series": "Seri", "Forecast": "Tahmin", "Lower": "Alt Sınır", "Upper": "Üst Sınır",
    "Method": "Yöntem", "Best": "En İyi Yöntem", "Son_Dönem": "Son 4 Hafta", "Değişim_%": "Değişim %",
    "Yöntem": "Yöntem", "Tahmin": "Tahmin", "Previous": "Önceki Dönem", "Last": "Son Dönem", "Change_%": "Değişim %",
    # danışman araç çıktılarındaki alanlar
    "revenue": "Ciro (₺)", "quantity": "Adet", "invoices": "Fatura", "customers": "Müşteri", "products": "Ürün",
    "categories": "Kategori", "avg_discount": "Ort. iskonto", "date_min": "Başlangıç", "date_max": "Bitiş",
    "note": "Not", "product": "Ürün", "stock_code": "Ürün Kodu", "dna": "DNA", "segment": "Segment", "item": "Öğe",
    "category": "Kategori", "message": "Mesaj", "customer_id": "Müşteri", "city": "Şehir", "cluster": "Küme",
    "recency_days": "Son alımdan beri (gün)", "orders": "Sipariş", "avg_order_value": "Ort. sipariş tutarı",
    "categories_bought": "Alınan kategori sayısı", "rfm_score": "RFM skoru", "action": "Önerilen yaklaşım",
    "branches": "Alt hesaplar", "top_categories": "En çok alınan kategoriler", "periods": "Dönemler",
    "thresholds": "Eşikler", "validation": "Doğrulama", "validation_wape_%": "Doğrulama WAPE %",
    "Opportunities": "Fırsat Sayısı", "Top_Products": "Önerilecek Ürünler", "Median_Score": "Medyan Puan",
    "Customer_Revenue": "Müşteri Yıllık Cirosu (₺)", "Test_WAPE_%": "Test WAPE %",
    "table": "Tablo", "dna_summary": "DNA özeti", "actions": "Aksiyonlar", "total": "Toplam",
    "category_share_%": "Kategori payı %", "similar_products_bought": "Alınan benzer ürünler",
    "n_similar_products_bought": "Benzer ürün sayısı", "product_buyers": "Ürünü alan müşteri", "already_bought": "Daha önce almış",
    "days_since_last": "Son alımdan beri (gün)", "Order_Days": "Sipariş Günü", "Typical_Gap": "Tipik Aralık (gün)",
    "Last_Purchase": "Son Sipariş", "Delay_Ratio": "Gecikme (kat)", "Expected_By": "Beklenen Sipariş",
    "Customer_Share_%": "Müşteri Payı %", "Company_Share_%": "Şirket Payı %", "Difference_pp": "Fark (puan)",
}


MONTHS_TR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
MONTHS_TR_LONG = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]


def month_tr(period, long: bool = False) -> str:
    """'2020-10' veya Period -> 'Eki' (long=True: 'Ekim')."""
    m = int(str(period)[5:7])
    return (MONTHS_TR_LONG if long else MONTHS_TR)[m - 1]


def tr_df(df):
    """DataFrame sütun ve indeks adlarını Türkçeye çevirir (kopya döndürür)."""
    out = df.rename(columns=lambda c: COLS.get(c, c))
    try:
        if out.index.name:
            out.index = out.index.rename(COLS.get(out.index.name, out.index.name))
    except Exception:
        pass
    return out


def localize_plotly_express():
    """px grafiklerinde eksen/lejant etiketlerini otomatik Türkçeleştirir."""
    import functools
    import plotly.express as px
    for name in ["bar", "line", "scatter", "scatter_3d", "area", "histogram", "box"]:
        fn = getattr(px, name)
        if getattr(fn, "_bp_localized", False):
            continue

        @functools.wraps(fn)
        def wrapped(*a, _fn=fn, **k):
            k["labels"] = {**COLS, **(k.get("labels") or {})}
            return _fn(*a, **k)
        wrapped._bp_localized = True
        setattr(px, name, wrapped)


# ---------------------------------------------------------------------------
GLOSSARY = {
    "ciro": "Satışlardan elde edilen toplam gelir. Bu projede her satır için Adet × Net Fiyat ile hesaplanır.",
    "iskonto": "Liste fiyatından yapılan indirim. Oran olarak: 1 − (Net Fiyat ÷ Birim Fiyat).",
    "pareto": "80/20 kuralı: sonuçların büyük kısmının az sayıda kaynaktan (ör. birkaç büyük müşteriden) gelmesi.",
    "yoğunlaşma": "Cironun az sayıda müşteri, ürün veya kategoride toplanma derecesi. Yüksekse bağımlılık riski artar.",
    "değişkenlik": "Değişkenlik katsayısı (CV) = aylık satışların standart sapması ÷ ortalaması. Yüksekse talep dalgalıdır.",
    "medyan": "Değerler küçükten büyüğe sıralandığında tam ortada kalan değer. Uç değerlerden etkilenmez.",
    "silhouette": "Kümelerin ne kadar iyi ayrıştığını ölçer (−1 ile 1 arası). 1'e yakın = net ayrışmış kümeler.",
    "küme": "Birbirine benzeyen müşterilerin algoritma tarafından oluşturulan grubu.",
    "segment": "Benzer satın alma davranışı gösteren müşteri grubu (ör. En Değerli Müşteriler, Riskli Müşteriler).",
    "sepet": "Tek bir faturada birlikte satın alınan ürünlerin tamamı.",
    "kaldıraç": "Lift: A'yı alanların B'yi alma olasılığının, B'nin genel alınma olasılığına oranı. 1'den büyükse birliktelik tesadüf değil.",
    "çapraz satış": "Bir ürünü alan müşteriye onu tamamlayan başka bir ürünü de satmak.",
    "trend": "Son 3 aydaki satış adedinin önceki 3 aya göre değişimi.",
    "performans puanı": "0–100 arası göreli ürün puanı: ciro %35, müşteri sayısı %25, son 3 ay trendi %20, talep istikrarı %20. Kâr veya stok bilgisi içermez.",
    "geriye dönük test": "Modeli geçmiş bir tarihe kadarki veriyle kurup, sonrasında gerçekte olanlarla karşılaştırarak başarısını ölçmek.",
    "isabet": "Önerilen müşterinin ürünü test döneminde gerçekten satın alması.",
    "uyuyan": "Belirli bir süredir (60+ gün) hiç alışveriş yapmamış müşteri.",
    "geri kazanım": "Uzun süredir alışveriş yapmayan müşteriyi yeniden aktif hale getirmeye yönelik arama veya teklif.",
    "üstel düzleştirme": "Son gözlemlere daha fazla ağırlık veren basit tahmin yöntemi (SES).",
    "naif": "En basit tahmin: gelecek hafta, geçen haftayla aynı olur varsayımı. Diğer yöntemler bununla kıyaslanır.",
    "gradient boosting": "Çok sayıda küçük karar ağacını art arda eğiterek tahmin yapan makine öğrenmesi yöntemi.",
    "tahmin aralığı": "Gerçek değerin %80 olasılıkla düşmesi beklenen alt-üst sınır bandı.",
    "dil modeli": "Büyük dil modeli (LLM): doğal dili anlayıp yanıt üretebilen yapay zekâ modeli (ör. Ollama ile çalışan Qwen, Llama, Gemma).",
    "api anahtarı": "Uygulamanın dil modeli servisine bağlanmak için kullandığı gizli erişim şifresi.",
    "belge türü": "Fatura numarasının ilk üç harfi (BAD, BSR, DPE, DPF, MUH); farklı satış kanallarını/kayıt türlerini gösterir.",
}
GLOSSARY_EXACT = {
    "RFM": "Recency – Frequency – Monetary: son alımdan bu yana geçen gün, sipariş sayısı ve ciro. Müşteri değerini ölçer.",
    "K-Means": "Benzer müşterileri otomatik olarak k adet gruba ayıran kümeleme algoritması.",
    "CV": "Değişkenlik katsayısı = standart sapma ÷ ortalama. Talebin ne kadar dalgalı olduğunu gösterir.",
    "HHI": "Herfindahl-Hirschman Endeksi: payların karelerinin toplamı (0–10.000). Yüksekse ciro birkaç üründe toplanmış.",
    "WAPE": "Ağırlıklı mutlak yüzde hata = toplam mutlak hata ÷ toplam gerçekleşen satış. Düşük = daha iyi tahmin.",
    "MAPE": "Ortalama mutlak yüzde hata. Küçük satışlı haftalarda şişebilir; WAPE daha güvenilir.",
    "MAE": "Ortalama mutlak hata: tahminin gerçekten ortalama kaç adet saptığı.",
    "RMSE": "Kök ortalama kare hata: büyük hataları daha çok cezalandıran hata ölçüsü.",
    "Destek": "Support: iki ürünü birlikte içeren sepetlerin tüm sepetlere oranı.",
    "Güven": "Confidence: A ürününü alanların B ürününü de alma oranı.",
    "B2B": "İşletmeden işletmeye satış (ör. yayıncının kitapçılara satışı).",
    "DNA": "Ürünün veya müşterinin satış davranışına göre atandığı profil grubu.",
    "LLM": "Büyük dil modeli: doğal dili anlayıp yanıt üretebilen yapay zekâ modeli.",
}

import re as _re


def _tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def _stem_pattern(stem: str) -> str:
    first = stem[0]
    up = {"i": "İ", "ı": "I"}.get(first, first.upper())
    return f"[{_re.escape(first)}{_re.escape(up)}]" + _re.escape(stem[1:])


_SORTED = sorted(GLOSSARY, key=len, reverse=True)
_SORTED_EXACT = sorted(GLOSSARY_EXACT, key=len, reverse=True)
_PATTERN = _re.compile(
    r"(?<![\w-])(" + "|".join([_re.escape(t) for t in _SORTED_EXACT] + [_stem_pattern(s) for s in _SORTED]) + r")([\wçğıöşüÇĞİÖŞÜ']*)")


def _lookup(matched: str):
    if matched in GLOSSARY_EXACT:
        return matched, GLOSSARY_EXACT[matched]
    key = _tr_lower(matched)
    return (key, GLOSSARY[key]) if key in GLOSSARY else (None, None)


def term_help(text: str):
    """Metindeki ilk sözlük teriminin açıklaması (tablo sütun başlıkları için)."""
    m = _PATTERN.search(str(text))
    return _lookup(m.group(1))[1] if m else None


def glossify(html_text: str, once: bool = True) -> str:
    """HTML metnindeki sözlük terimlerini üzerine gelince açıklama gösteren etiketlerle sarar."""
    seen = set()
    parts = _re.split(r"(<[^>]+>)", html_text)
    inside_term = 0
    for i, part in enumerate(parts):
        if part.startswith("<"):
            if 'class="bp-term"' in part:
                inside_term += 1
            elif part.startswith("</span") and inside_term:
                inside_term -= 1
            continue
        if inside_term or not part.strip():
            continue

        def repl(m):
            key, tip = _lookup(m.group(1))
            if not tip or (once and key in seen):
                return m.group(0)
            seen.add(key)
            return f'<span class="bp-term" tabindex="0" data-tip="{html.escape(tip, quote=True)}">{m.group(0)}</span>'
        parts[i] = _PATTERN.sub(repl, part)
    return "".join(parts)


TOOLTIP_CSS = f"""
<style>
.bp-term {{ border-bottom: 1.5px dotted currentColor; cursor: help; position: relative; text-decoration: none; }}
.bp-term:hover::after, .bp-term:focus::after {{
    content: attr(data-tip); position: absolute; left: 50%; bottom: calc(100% + 10px); transform: translateX(-50%);
    width: max-content; max-width: 280px; white-space: normal; text-align: left;
    background: {P['navy']}; color: #fff; padding: 9px 12px; border-radius: 10px;
    font-size: 12.5px; font-weight: 500; line-height: 1.4; letter-spacing: normal; text-transform: none;
    box-shadow: 0 12px 28px -10px rgba(20,22,58,.55); z-index: 9999; pointer-events: none;
}}
.bp-term:hover::before, .bp-term:focus::before {{
    content: ""; position: absolute; left: 50%; bottom: calc(100% + 4px); transform: translateX(-50%);
    border: 6px solid transparent; border-top-color: {P['navy']}; z-index: 9999;
}}
.bp-hero .bp-term:hover::after {{ background: #fff; color: {P['ink']}; }}
.bp-hero .bp-term:hover::before {{ border-top-color: #fff; }}
</style>
"""


# ---------------------------------------------------------------------------
_ICONS = {
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "receipt": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "trend_up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trend_down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "hexagon": '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>',
    "card": '<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>',
    "bars": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
}


def icon(name: str, size: int = 16) -> str:
    body = _ICONS.get(name)
    if body is None:
        return _e(name)
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


# ---------------------------------------------------------------------------
_UNITS = {1: ("bir", "i"), 2: ("iki", "i"), 3: ("üç", "ü"), 4: ("dört", "ü"), 5: ("beş", "i"), 6: ("altı", "ı"),
          7: ("yedi", "i"), 8: ("sekiz", "i"), 9: ("dokuz", "u")}
_TENS = {1: ("on", "u"), 2: ("yirmi", "i"), 3: ("otuz", "u"), 4: ("kırk", "ı"), 5: ("elli", "i"), 6: ("altmış", "ı"),
         7: ("yetmiş", "i"), 8: ("seksen", "i"), 9: ("doksan", "ı")}


def _last_word(n: int):
    if n == 0:
        return ("sıfır", "ı")
    if n % 10:
        return _UNITS[n % 10]
    if n % 100:
        return _TENS[(n % 100) // 10]
    return ("yüz", "ü") if n % 1000 else ("bin", "i")


def pct(value: float, decimals: int = 0, case: str = "poss") -> str:
    """Yüzdeyi Türkçe ekiyle yazar. case: 'poss' (%83'ü / %20'si) veya 'poss_acc' (%83'ünü / %20'sini)."""
    if decimals and round(value, decimals) == round(value):  
        decimals = 0
    txt = f"{value:.{decimals}f}".replace(".", ",")
    digits = txt.split(",")[-1] if decimals else txt
    word, vowel = _last_word(int(digits.lstrip("0") or 0) if decimals else int(round(value)))
    ends_vowel = word[-1] in "aeıioöuü"
    suffix = ("s" if ends_vowel else "") + vowel
    if case == "poss_acc":
        suffix += "n" + vowel
    return f"%{txt}'{suffix}"


# ---------------------------------------------------------------------------
NAV_CSS = f"""
<style>
/* başlıklardaki Streamlit bağlantı simgesini gizle */
.bp-section [data-testid="stHeaderActionElements"], .bp-header [data-testid="stHeaderActionElements"],
.bp-section h3 a, .bp-header h1 a {{ display:none !important; }}
section[data-testid="stSidebar"] .bp-navgroup {{
    font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: {P['muted']};
    margin: 14px 4px 4px;
}}
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%; justify-content: flex-start; text-align: left; border: none; background: transparent;
    color: {P['text']}; font-weight: 500; padding: 8px 12px; border-radius: 10px; min-height: 0;
}}
section[data-testid="stSidebar"] .stButton > button p {{ font-size: 14.5px; text-align: left; }}
section[data-testid="stSidebar"] .stButton > button div {{ justify-content: flex-start; }}
section[data-testid="stSidebar"] .stButton > button:hover {{ background: {P['bg']}; color: {P['ink']}; }}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{
    background: {P['indigo_50']}; color: {P['indigo_700']}; font-weight: 700;
    box-shadow: inset 3px 0 0 {P['indigo']};
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p {{ font-weight: 700; }}
section[data-testid="stSidebar"] .stButton {{ margin-bottom: -10px; }}

.bp-insight {{ display:flex; gap:10px; align-items:flex-start; margin: -4px 2px 18px; color:{P['text']};
               font-size: 14px; line-height: 1.45; }}
.bp-insight .tag {{ flex: none; font-size: 11px; font-weight: 700; letter-spacing:.06em; text-transform: uppercase;
                    color: {P['indigo_700']}; background:{P['indigo_50']}; border-radius: 6px; padding: 2px 7px; margin-top:1px; }}
.bp-insight.risk .tag {{ color:{P['red_600']}; background:{P['red_50']}; }}

.bp-action {{ background:#fff; border:1px solid {P['line']}; border-radius:16px; padding:14px 16px; height:100%;
              box-shadow: 0 8px 24px -16px rgba(20,22,58,.18); }}
.bp-action .t {{ font-size:12px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:{P['muted']}; }}
.bp-action .v {{ font-size:1.35rem; font-weight:800; color:{P['ink']}; margin:4px 0 2px; }}
.bp-action .d {{ font-size:13.5px; color:{P['text']}; line-height:1.4; }}
.bp-action.risk {{ box-shadow: inset 4px 0 0 {P['red']}, 0 8px 24px -16px rgba(20,22,58,.18); }}
.bp-action.warn {{ box-shadow: inset 4px 0 0 {P['amber']}, 0 8px 24px -16px rgba(20,22,58,.18); }}
.bp-action.info {{ box-shadow: inset 4px 0 0 {P['indigo']}, 0 8px 24px -16px rgba(20,22,58,.18); }}

.bp-steps {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin: 2px 0 14px; }}
.bp-step {{ font-size:13px; padding:6px 12px; border-radius:999px; background:#fff; border:1px solid {P['line']};
            color:{P['muted']}; font-weight:600; }}
.bp-step.on {{ background:{P['indigo']}; border-color:{P['indigo']}; color:#fff; }}
.bp-step.done {{ color:{P['indigo_700']}; border-color:#DCDFFF; background:{P['indigo_50']}; }}
.bp-arrow {{ color:{P['muted']}; font-size:13px; }}

.bp-ref {{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:600; color:{P['muted']};
           background:#fff; border:1px dashed {P['line']}; border-radius:999px; padding:5px 12px; }}

@media (max-width: 760px) {{
  .bp-header {{ flex-direction: column; align-items: flex-start; }}
  .bp-header h1 {{ font-size: 1.6rem; }}
  .bp-kpis {{ grid-template-columns: 1fr 1fr; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


def insight(text_html: str, tag: str = "Ne görüyoruz?", risk: bool = False) -> str:
    """Grafiğin altında 'bu grafik ne anlatıyor' cümlesi."""
    return f'<div class="bp-insight {"risk" if risk else ""}"><span class="tag">{_e(tag)}</span><span>{glossify(text_html)}</span></div>'


def action_card(title: str, value: str, text_html: str, tone: str = "info") -> str:
    return (f'<div class="bp-action {tone}"><div class="t">{_e(title)}</div><div class="v">{_e(value)}</div>'
            f'<div class="d">{glossify(text_html)}</div></div>')


def steps(labels: list[str], active: int) -> str:
    parts = []
    for i, l in enumerate(labels):
        cls = "on" if i == active else ("done" if i < active else "")
        parts.append(f'<span class="bp-step {cls}">{i + 1}. {_e(l)}</span>')
    return '<div class="bp-steps">' + '<span class="bp-arrow">→</span>'.join(parts) + "</div>"


REF_NOTE = "Geçmiş veriye dayalı karar simülasyonu · referans tarih 31.12.2020"
