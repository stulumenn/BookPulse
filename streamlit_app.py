"""BookPulse – Yapay Zekâ Destekli B2B Satış Zekâsı Platformu (Streamlit).

Çalıştırma:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Streamlit Community Cloud: ana dosya olarak streamlit_app.py seçilir.
"""
from __future__ import annotations

import inspect
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bookpulse import basket, config, customer_dna, data, forecast, next_calls, portfolio, ui
from bookpulse.advisor import (PROVIDERS, SalesAdvisor, error_detail, ollama_settings, ollama_status, ping_provider,
                               provider_key, setup_guide)
from bookpulse.engine import BookPulse
from bookpulse.ui import P

_LOGO = config.ROOT / "assets" / "logo.png"
try:
    from PIL import Image
    _ICON = Image.open(_LOGO)
except Exception:
    _ICON = None
st.set_page_config(page_title="BookPulse", page_icon=_ICON, layout="wide")
ui.setup()

NAV = [
    ("Genel Bakış", ["Yönetici Paneli"]),
    ("Ürün Analitiği", ["Portföy Optimizasyonu"]),
    ("Müşteri Analitiği", ["Müşteri DNA"]),
    ("Satış ve Talep", ["Sepet Analizi", "Talep Tahmini"]),
    ("Karar Destek", ["Sıradaki 10 Arama", "Yapay Zekâ Satış Danışmanı"]),
]
PAGES = [p_ for _, items in NAV for p_ in items]
STAR, VOLATILE = "Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep"
PRIORITY_ORDER = {"Hemen Ara": 0, "Geri Kazanım Araması": 1, "Teklif Sun": 2, "Takip Et": 3}
METHOD_TR = {"Naive": "Naif (geçen hafta)", "MA_4": "4 haftalık ortalama", "SES": "Üstel düzleştirme",
             "GBM": "Gradient boosting", "Seri bazında seçim": "Seri bazında seçim"}


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def tl(x: float) -> str:
    return f"{num(x)} ₺"


def tl_m(x: float) -> str:
    return f"{x / 1e6:,.1f} M₺".replace(".", ",")


def pct1(x: float) -> str:
    return f"%{x:.1f}".replace(".", ",")


def spct(x: float, decimals: int = 0) -> str:
    """İşaretli Türkçe yüzde: +%29 / −%71."""
    sign = "+" if x > 0 else ("−" if x < 0 else "")
    return f"{sign}%{abs(x):.{decimals}f}".replace(".", ",")


def dtr(d) -> str:
    """'2020-12-31' → '31.12.2020'."""
    return pd.Timestamp(d).strftime("%d.%m.%Y")


def md(html: str, where=st):
    where.markdown(html, unsafe_allow_html=True)


def rerun():
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


def goto(page: str, **state):
    """Sayfa değiştirir; verilen seçimleri ilgili kalıcı anahtarlara ve widget'lara uygular."""
    if state.get("selected_product") is not None:
        state["_calls_dna_sync"] = True
    if state.get("selected_customer"):
        state["_cust_id"] = state["selected_customer"]
    for k_ in list(state):
        if k_.startswith("_"):
            st.session_state.pop("w" + k_, None)     
    st.session_state.pop("w_calls_dna", None) if state.get("_calls_dna_sync") else None
    st.session_state.update(state)
    st.session_state["page"] = page
    rerun()


def keep_select(where, label, options, key, default=None, **kw):
    options = list(options)
    if not options:
        return None
    stored = st.session_state.get("_" + key, default)
    idx = options.index(stored) if stored in options else 0
    v = where.selectbox(label, options, index=idx, key="w_" + key, **kw)
    st.session_state["_" + key] = v
    return v


def keep_multi(where, label, options, key, **kw):
    stored = [x for x in st.session_state.get("_" + key, []) if x in options]
    if "placeholder" in inspect.signature(st.multiselect).parameters:
        kw.setdefault("placeholder", "Tümü (seçmek için tıklayın)")
    v = where.multiselect(label, options, default=stored, key="w_" + key, **kw)
    st.session_state["_" + key] = v
    return v


def keep_slider(where, label, key, default, **kw):
    v = where.slider(label, value=st.session_state.get("_" + key, default), key="w_" + key, **kw)
    st.session_state["_" + key] = v
    return v


def keep_text(where, label, key, default="", **kw):
    v = where.text_input(label, value=st.session_state.get("_" + key, default), key="w_" + key, **kw)
    st.session_state["_" + key] = v
    return v


def _stretch_kw(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and params["width"].default == "stretch":
        return {}
    if "width" in params and isinstance(params["width"].default, str):
        return {"width": "stretch"}
    if "use_container_width" in params and params["use_container_width"].default in (False, None):
        return {"use_container_width": True}
    return {}


_CHART_KW = _stretch_kw(st.plotly_chart)
_TABLE_KW = _stretch_kw(st.dataframe)
_ROW_SELECT = "on_select" in inspect.signature(st.dataframe).parameters
_PLOT_SELECT = "on_select" in inspect.signature(st.plotly_chart).parameters


def chart(where, fig, height: int | None = None, note: str | None = None, risk: bool = False):
    """Grafik + altında 'Ne görüyoruz?' yorumu."""
    if height:
        fig.update_layout(height=height)
    fig.update_layout(legend_title_text="")
    where.plotly_chart(fig, theme=None, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                       **_CHART_KW)
    if note:
        md(ui.insight(note, risk=risk), where)


def show(where, df, **kw):
    """Küçük tablolar: Türkçe başlıklar ve terim açıklamaları."""
    t = ui.tr_df(df)
    cfg = dict(kw.pop("column_config", None) or {})
    for col in list(t.columns) + ([t.index.name] if t.index.name else []):
        tip = ui.term_help(col)
        if tip and col not in cfg:
            cfg[col] = st.column_config.Column(help=tip)
    return where.dataframe(t, column_config=cfg or None, **{**_TABLE_KW, **kw})


def _excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=df.index.name is not None, sheet_name="BookPulse")
        return buf.getvalue()
    except Exception:
        return None


def table(df: pd.DataFrame, key: str, page_size: int = 25, column_config=None, hide_index: bool = True,
          height: int | None = None, filename: str = "bookpulse"):
    """Büyük tablolar: arama, sıralama (başlığa tıklayarak), sayfalama, sabit başlık, CSV/Excel dışa aktarma."""
    if df is None or df.empty:
        st.info("Gösterilecek kayıt yok.")
        return
    t = ui.tr_df(df)
    c1, c2, c3, c4 = st.columns([3, 1.1, 1, 1])
    q = c1.text_input("Tabloda ara", key=f"q_{key}", placeholder="ürün, müşteri, kategori…",
                      label_visibility="collapsed") if "label_visibility" in inspect.signature(st.text_input).parameters \
        else c1.text_input("Tabloda ara", key=f"q_{key}")
    if q:
        mask = pd.Series(False, index=t.index)
        for col in t.columns:
            if t[col].dtype == object or str(t[col].dtype).startswith(("str", "string")):
                mask |= t[col].astype(str).str.contains(q, case=False, regex=False, na=False)
        if t.index.dtype == object:
            mask |= t.index.astype(str).str.contains(q, case=False, regex=False)
        t = t[mask]
    pages = max(1, int(np.ceil(len(t) / page_size)))
    page = int(c2.number_input("Sayfa", 1, pages, 1, key=f"p_{key}")) if pages > 1 else 1
    c2.caption(f"{num(len(t))} kayıt · {pages} sayfa")
    view = t.iloc[(page - 1) * page_size: page * page_size]
    c3.download_button("CSV", t.to_csv(index=not hide_index).encode("utf-8-sig"), f"{filename}.csv", "text/csv",
                       key=f"csv_{key}")
    xb = _excel_bytes(t if not hide_index else t.reset_index(drop=True))
    if xb:
        c4.download_button("Excel", xb, f"{filename}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xls_{key}")
    cfg = dict(column_config or {})
    for col in view.columns:
        tip = ui.term_help(col)
        if tip and col not in cfg:
            cfg[col] = st.column_config.Column(help=tip)
    kw = {**_TABLE_KW, "hide_index": hide_index, "column_config": cfg or None}
    if height:
        kw["height"] = height
    st.dataframe(view, **kw)


def ref_chip() -> str:
    return f'<span class="bp-ref">{ui.REF_NOTE}</span>'


@st.cache_resource(show_spinner="Veri yükleniyor ve modeller hazırlanıyor…")
def get_engine() -> BookPulse:
    return BookPulse(data.load_clean())


@st.cache_data(show_spinner="Geriye dönük test çalışıyor (≈30 sn)…")
def run_backtest(n_products: int) -> dict:
    res = next_calls.backtest(get_engine().df, n_products=n_products)
    return {"table": res["table"], "n_products": res["n_products"], "cutoff": res["cutoff"]}


def stored_backtest() -> dict | None:
    p_ = config.DATA_DIR / "backtest_results.json"
    if not p_.exists():
        return None
    d = json.loads(p_.read_text())
    d["table"] = pd.DataFrame(d["table"]).set_index("Model")
    return d


@st.cache_data(show_spinner="Toplu arama listesi oluşturuluyor…")
def run_batch(dnas: tuple, n_products: int, n: int) -> pd.DataFrame:
    bp = get_engine()
    p_ = bp.portfolio
    codes = p_[p_["DNA"].isin(dnas)].nlargest(n_products, "Revenue").index
    return bp.calls.batch(codes, n=n)


@st.cache_data(show_spinner=False)
def _missing_orders_cached(_n: int) -> pd.DataFrame:
    return customer_dna.missing_orders(get_engine().df)


def missing_orders(bp: BookPulse) -> pd.DataFrame:
    return _missing_orders_cached(len(bp.df))


def running_on_cloud() -> bool:
    """Streamlit Community Cloud'da mı çalışıyoruz? (uygulama /mount/src altında, kullanıcı appuser)"""
    return os.path.isdir("/mount/src") or os.getenv("HOME", "") == "/home/appuser" or bool(os.getenv("BOOKPULSE_CLOUD"))


def data_available() -> bool:
    return any(p_.exists() for p_ in [config.CLEAN_FILE, config.RAW_XLSX])


# ===================================================================================
def sidebar() -> str:
    md(ui.brand(), st.sidebar)
    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = PAGES[0]
    current = st.session_state["page"]
    for group, items in NAV:
        md(f'<div class="bp-navgroup">{group}</div>', st.sidebar)
        for item in items:
            if st.sidebar.button(item, key=f"nav_{item}", type="primary" if item == current else "secondary"):
                if item != current:
                    st.session_state["last_page"] = current
                    goto(item)
    st.sidebar.divider()
    md(f'<div style="font-size:12.5px;color:{P["muted"]};line-height:1.5">'
       + ui.glossify("Veri: 2020 B2B satış işlemleri (referans tarih 31.12.2020)<br>"
                     "Ciro = Adet × Net Fiyat (hesaplanan)<br>"
                     "Tutarlar TL (₺) varsayılmıştır; para birimi veride belirtilmez.") + "</div>", st.sidebar)
    return st.session_state["page"]


# ===================================================================================
def _auto_insights(df: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    out = {}
    if len(monthly) >= 2:
        hi, lo = monthly.loc[monthly["Ciro"].idxmax()], monthly.loc[monthly["Ciro"].idxmin()]
        half = len(monthly) // 2
        h1, h2 = monthly["Ciro"].iloc[:half].sum(), monthly["Ciro"].iloc[half:].sum()
        out["monthly"] = (f"Ciro en yüksek <b>{ui.MONTHS_TR_LONG[ui.MONTHS_TR.index(hi['Month'])]}</b> ayında "
                          f"({tl_m(hi['Ciro'])}), en düşük <b>{ui.MONTHS_TR_LONG[ui.MONTHS_TR.index(lo['Month'])]}</b> "
                          f"ayında ({tl_m(lo['Ciro'])}). Dönemin ikinci yarısı ilk yarısına göre "
                          f"<b>{spct(100 * (h2 - h1) / max(h1, 1))}</b>.")
    cat = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    if len(cat):
        top3 = cat.head(3)
        out["category"] = (f"İlk üç kategori ({', '.join(top3.index)}) cironun "
                           f"<b>{ui.pct(100 * top3.sum() / cat.sum(), 0, 'poss_acc')}</b> oluşturuyor.")
    if df["Month"].nunique() >= 6:
        months = sorted(df["Month"].unique())
        q = df.groupby(["Category", "Month"])["Quantity"].sum().unstack(fill_value=0)
        ch = pd.DataFrame({"prev": q[months[-6:-3]].sum(axis=1), "last": q[months[-3:]].sum(axis=1)})
        ch = ch[ch["prev"] >= 300]
        if len(ch):
            ch["pct"] = 100 * (ch["last"] - ch["prev"]) / ch["prev"]
            out["trend_up"], out["trend_down"] = ch["pct"].idxmax(), ch["pct"].idxmin()
            out["trend_up_pct"], out["trend_down_pct"] = ch["pct"].max(), ch["pct"].min()
    city = df.groupby("City")["Revenue"].sum().sort_values(ascending=False)
    if len(city) > 1:
        out["city"] = (f"<b>{city.index[0]}</b> tek başına cironun "
                       f"<b>{ui.pct(100 * city.iloc[0] / city.sum(), 0, 'poss_acc')}</b> oluşturuyor; ikinci sıradaki "
                       f"{city.index[1]} {ui.pct(100 * city.iloc[1] / city.sum())} payında.")
    return out


def page_executive(bp: BookPulse):
    full = bp.df
    md(ui.page_header("Yönetici Paneli", "Satış performansının özeti ve dikkat gerektiren durumlar. Filtreler tüm sayfa "
                                         "boyunca geçerlidir ve sayfa değiştirince korunur.",
                      eyebrow="Genel Bakış", chip=f"{dtr(bp.kpis['date_min'])} – {dtr(bp.kpis['date_max'])}"))

    months_all = sorted(full["Month"].unique())
    labels = [ui.MONTHS_TR_LONG[m.month - 1] for m in months_all]
    f1, f2, f3, f4 = st.columns([2.2, 2, 2, 0.9])
    rng = st.session_state.get("_ex_months", (labels[0], labels[-1]))
    start, end = f1.select_slider("Ay aralığı", options=labels, value=rng, key="w_ex_months")
    st.session_state["_ex_months"] = (start, end)
    cities = keep_multi(f2, "Şehir", sorted(full["City"].unique()), "ex_city")
    cats = keep_multi(f3, "Kategori", sorted(full["Category"].unique()), "ex_cat")
    f4.write("")
    if f4.button("Temizle"):
        for k_ in ["_ex_months", "_ex_city", "_ex_cat"]:
            st.session_state.pop(k_, None)
        rerun()
    sel_months = months_all[labels.index(start): labels.index(end) + 1]
    df = full[full["Month"].isin(sel_months)]
    if cities:
        df = df[df["City"].isin(cities)]
    if cats:
        df = df[df["Category"].isin(cats)]
    if df.empty:
        st.warning("Seçilen filtrelerle eşleşen satış bulunamadı. Filtreleri genişletin veya **Temizle**'ye basın.")
        return
    k = data.kpis(df)
    cust = df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
    share = cust.cumsum() / cust.sum() * 100
    top20 = float(share.iloc[max(int(len(share) * 0.2) - 1, 0)])

    md(ui.kpi_cards([
        {"label": "Ciro", "value": tl_m(k["revenue"]), "icon": "card", "tone": "indigo",
         "note": "Adet × net fiyat (hesaplanan)"},
        {"label": "Sipariş (fatura)", "value": num(k["invoices"]), "icon": "receipt", "tone": "indigo",
         "note": f"ortalama {tl(k['revenue'] / k['invoices'])} / fatura"},
        {"label": "Aktif müşteri", "value": num(k["customers"]), "icon": "users", "tone": "indigo",
         "note": "dönemde en az bir alım yapan"},
        {"label": "Satılan adet", "value": num(k["quantity"]), "icon": "book", "tone": "indigo",
         "note": f"{num(k['products'])} farklı ürün"},
        {"label": "Müşteri yoğunlaşması", "value": f"%{top20:.0f}",
         "icon": "target", "tone": "red" if top20 >= 80 else "indigo",
         "note": "cironun en büyük %20 müşteriden gelen payı"},
    ]))

    monthly = df.groupby("Month").agg(Ciro=("Revenue", "sum"), Müşteri=("CustomerID", "nunique")).reset_index()
    monthly["Month"] = monthly["Month"].map(ui.month_tr)
    ins = _auto_insights(df, monthly)
    mo = missing_orders(bp)
    declining = bp.portfolio[bp.portfolio["Action"] == "İzle (Düşüşte)"]
    risk = bp.customers[bp.customers["Segment"].isin(["Riskli Müşteriler", "Riskli Değerli Müşteriler"])]
    md(ui.section("Dikkat gerektirenler", "öncelikli incelenmesi önerilen durumlar"))
    a1, a2, a3, a4 = st.columns(4)
    md(ui.action_card("Kaçan siparişler", f"{num(len(mo))} müşteri",
                      f"Normal sipariş aralığının 2 katından uzun süredir sessiz; yıllık cirosu {tl_m(mo['Revenue'].sum())}.",
                      "risk"), a1)
    if a1.button("Listeyi aç", key="act_mo"):
        goto("Müşteri DNA", cust_tab="Kaçan Sipariş Dedektörü")
    md(ui.action_card("Risk altındaki değerli müşteriler", f"{num(len(risk))} müşteri",
                      f"Eskiden sık alıp uzun süredir almayanlar; yıllık cirosu {tl_m(risk['Monetary'].sum())}.", "risk"), a2)
    if a2.button("Segmenti aç", key="act_risk"):
        goto("Müşteri DNA", cust_tab="Segmentler")
    if "trend_down" in ins:
        md(ui.action_card("En çok düşen kategori", ins["trend_down"],
                          f"Son 3 ayda adet <b>{spct(ins['trend_down_pct'])}</b>; en hızlı büyüyen "
                          f"<b>{ins['trend_up']}</b> ({spct(ins['trend_up_pct'])}).", "warn"), a3)
    if a3.button("Talep tahminine git", key="act_fc"):
        goto("Talep Tahmini")
    md(ui.action_card("Düşüşteki güçlü ürünler", f"{num(len(declining))} ürün",
                      f"Çok satan ve talebi düzenli ürünlerde son 3 ayda belirgin düşüş; cirosu {tl_m(declining['Revenue'].sum())}.",
                      "warn"), a4)
    if a4.button("Ürünleri aç", key="act_decl"):
        goto("Portföy Optimizasyonu", _pf_action="İzle (Düşüşte)")

    md(ui.section("Satış trendi", "aylık ciro"))
    fig = px.bar(monthly, x="Month", y="Ciro", text_auto=".2s",
                 title="Aylık ciro (₺) – Covid kapanma ayları vurgulu", labels={"Month": "Ay", "Ciro": "Ciro (₺)"})
    fig.update_traces(marker_color=[P["coral"] if m in ("Nis", "Tem") else P["indigo"] for m in monthly["Month"]],
                      textposition="outside", textfont_color=P["muted"], cliponaxis=False,
                      hovertemplate="%{x}<br>Ciro: %{y:,.0f} ₺<extra></extra>")
    chart(st, fig, 380, ins.get("monthly"))

    l2, r2 = st.columns(2, gap="medium")
    cat = df.groupby("Category")["Revenue"].sum().nlargest(10).sort_values().reset_index()
    fig = px.bar(cat, x="Revenue", y="Category", orientation="h", title="Ciroya göre ilk 10 kategori",
                 labels={"Revenue": "Ciro (₺)", "Category": ""})
    fig.update_traces(marker_color=P["indigo"], hovertemplate="%{y}<br>Ciro: %{x:,.0f} ₺<extra></extra>")
    chart(l2, fig, 420, ins.get("category"))
    city = df.groupby("City")["Revenue"].sum().nlargest(10).sort_values().reset_index()
    fig = px.bar(city, x="Revenue", y="City", orientation="h", title="Ciroya göre ilk 10 şehir",
                 labels={"Revenue": "Ciro (₺)", "City": ""})
    fig.update_traces(marker_color=P["sky"], hovertemplate="%{y}<br>Ciro: %{x:,.0f} ₺<extra></extra>")
    chart(r2, fig, 420, ins.get("city"))

    pareto = pd.DataFrame({"Müşteri yüzdesi": np.arange(1, len(share) + 1) / len(share) * 100,
                           "Kümülatif ciro yüzdesi": share.to_numpy()})
    fig = px.area(pareto, x="Müşteri yüzdesi", y="Kümülatif ciro yüzdesi",
                  title="Müşteri yoğunlaşması (Pareto eğrisi)",
                  labels={"Müşteri yüzdesi": "Müşteriler (büyükten küçüğe, %)", "Kümülatif ciro yüzdesi": "Kümülatif ciro (%)"})
    fig.update_traces(line_color=P["indigo"], fillcolor="rgba(91,95,227,.14)",
                      hovertemplate="Müşterilerin %{x:.0f}%'i → cironun %{y:.0f}%'i<extra></extra>")
    fig.add_vline(x=20, line_dash="dash", line_color=P["muted"])
    chart(st, fig, 320, f"Müşterilerin en büyük %20'si cironun <b>{ui.pct(top20, 0, 'poss_acc')}</b> getiriyor: "
                        "birkaç büyük müşteriyi kaybetmek ciroyu belirgin etkiler.", risk=top20 >= 80)

    md(ui.section("En büyük müşteriler", "detay tablo"))
    top = df.groupby("CustomerID").agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"),
                                       Orders=("InvoiceID", "nunique"), City=("City", "first")).sort_values(
        "Revenue", ascending=False)
    top["Share_%"] = (100 * top["Revenue"] / top["Revenue"].sum()).round(2)
    top = top.join(bp.customers[["Segment"]]).round({"Revenue": 0})
    table(top, "ex_top", hide_index=False, filename="en_buyuk_musteriler")

    with st.expander("Veri temizliği ve varsayımlar"):
        items = [
            f"Fatura önekleri belge türü olarak ayrıldı; satış olarak dahil edilenler: <b>{', '.join(config.SALES_DOC_TYPES)}</b> (MUH hariç).",
            "Eksik veya sıfır miktarlı, sıfır ya da negatif net fiyatlı ve sembolik birim fiyatlı satırlar çıkarıldı (582 satır; "
            "ekibin ilk notebook'unda yalnızca 11 satır çıkarılmıştı – fark, belge türü ve fiyat kontrollerinden gelir).",
            "Kategori ve şehir adlarındaki boşluklar temizlendi; <code>A00001-01</code> gibi alt hesaplar ana müşteriye bağlandı.",
            "Excel'in sayıya veya tarihe çevirdiği ürün adları metne döndürüldü; birebir tekrar eden 6.213 satır korunup işaretlendi.",
            "Fiyatlar değiştirilmez; net fiyatı birim fiyattan yüksek satırlar yalnızca işaretlenir.",
            "Ciro = Adet × Net Fiyat. Para birimi ve fiyat alanlarının ticari tanımı veri sahibiyle doğrulanmalıdır.",
        ]
        md(ui.glossify("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"))


# ===================================================================================
def page_portfolio(bp: BookPulse):
    p, th = bp.portfolio, bp.thresholds
    md(ui.page_header("Portföy Optimizasyonu",
                      "Ürünler satış hacmi ve talep istikrarına göre beş DNA grubuna ayrılır; her ürüne göreli bir performans "
                      "puanı ve gerekçeli bir öneri atanır. Stok ve maliyet verisi olmadığından öneriler inceleme önerisidir.",
                      eyebrow="Ürün Analitiği"))
    summ = portfolio.dna_summary(p)
    icons = ["star", "zap", "target", "help", "clock"]
    tones = ["indigo", "coral", "blue", "red", "indigo"]
    md(ui.kpi_cards([{"label": d, "value": num(summ.loc[d, "Products"]) + " ürün", "icon": icons[i], "tone": tones[i],
                      "note": f"ciro payı <b>{pct1(summ.loc[d, 'Revenue_Share_%'])}</b>"}
                     for i, d in enumerate(portfolio.DNA_ORDER)]))

    t1, t2, t3, t4 = st.tabs(["Ürünler ve öneriler", "DNA haritası", "Ürün detayı", "Kategori yoğunlaşması"])
    with t1:
        f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.6, 1.4])
        dna = keep_select(f1, "DNA grubu", ["(tümü)"] + portfolio.DNA_ORDER, "pf_dna")
        acts = portfolio.action_summary(p)
        act = keep_select(f2, "Öneri", ["(tümü)"] + list(acts.index), "pf_action")
        cat = keep_select(f3, "Kategori", ["(tümü)"] + sorted(p["Category"].unique()), "pf_cat")
        min_perf = keep_slider(f4, "En düşük performans puanı", "pf_perf", 0, min_value=0, max_value=100, step=5)
        q = p[p["Performance_Score"] >= min_perf]
        if dna != "(tümü)":
            q = q[q["DNA"] == dna]
        if act != "(tümü)":
            q = q[q["Action"] == act]
        if cat != "(tümü)":
            q = q[q["Category"] == cat]
        md(ui.insight(f"Filtreyle <b>{num(len(q))}</b> ürün eşleşti; bu ürünler cironun "
                      f"<b>{ui.pct(100 * q['Revenue'].sum() / p['Revenue'].sum(), 1, 'poss_acc')}</b> oluşturuyor. "
                      "<i>Öneri gerekçesi</i> sütunu her ürünün neden bu öneriyi aldığını açıklar."))
        cols = ["Description", "Category", "DNA", "Action", "Action_Reason", "Performance_Score", "Revenue",
                "Total_Qty", "Trend", "Customers", "Peak_Month", "Days_Since_Sale"]
        out = q.sort_values("Revenue", ascending=False)[cols].round({"Revenue": 0, "Trend": 2})
        out = out.rename(columns={"Action": "Öneri", "Action_Reason": "Öneri gerekçesi"})
        table(out, "pf_list", hide_index=False, filename="portfoy_onerileri", column_config={
            "Performans Puanı": st.column_config.ProgressColumn("Performans Puanı", min_value=0, max_value=100, format="%.0f"),
            "Öneri gerekçesi": st.column_config.TextColumn("Öneri gerekçesi", width="large")})
        fig = px.bar(acts.reset_index().sort_values("Revenue"), x="Revenue", y="Action", orientation="h",
                     text="Products", title="Önerilere göre ciro (bar üzerindeki sayı: ürün adedi)",
                     labels={"Revenue": "Ciro (₺)", "Action": ""})
        fig.update_traces(marker_color=P["indigo"], texttemplate="%{text} ürün", textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>Ciro: %{x:,.0f} ₺<extra></extra>")
        fig.update_xaxes(range=[0, acts["Revenue"].max() * 1.18])
        top_act = acts.index[0]
        chart(st, fig, 360, f"Cironun en büyük kısmı <b>{top_act}</b> önerisindeki ürünlerden geliyor "
                            f"({pct1(acts.loc[top_act, 'Revenue_Share_%'])}). Ürünlerin büyük çoğunluğu ise düşük cirolu "
                            "uzun kuyrukta yer alıyor.")

    with t2:
        plot = p[p["DNA"] != "Yetersiz Satış Geçmişi"].reset_index()
        fig = px.scatter(plot, x="Total_Qty", y="CV", color="DNA", color_discrete_map=ui.DNA_COLORS,
                         log_x=True, opacity=0.45, render_mode="webgl",
                         title="Portföy DNA haritası: satış hacmi ve talep değişkenliği",
                         hover_data={"Description": True, "Category": True, "Action": True, "StockCode": True},
                         labels={"Total_Qty": "Toplam satış adedi (log ölçek)", "CV": "Talep değişkenliği (CV)"},
                         category_orders={"DNA": portfolio.DNA_ORDER})
        fig.add_vline(x=th["volume_threshold"], line_dash="dash", line_color=P["muted"])
        fig.add_hline(y=th["cv_threshold"], line_dash="dash", line_color=P["muted"])
        fig.update_traces(marker=dict(size=5))
        map_note = (f"Kesikli çizgiler eşikleri gösterir (hacim ≥ {th['volume_threshold']:.0f} adet, CV ≤ "
                    f"{th['cv_threshold']:.2f}). Sağ alt: <b>Çok Satan – Düzenli Talep</b> grubu. "
                    f"Satış geçmişi yetersiz {num(summ.loc['Yetersiz Satış Geçmişi', 'Products'])} ürün (3 aydan az veya 10 "
                    "adetten az satış) haritada gösterilmez; bu ürünlerin kategori dağılımı <i>Kategori yoğunlaşması</i> sekmesinde.")
        if _PLOT_SELECT:   
            fig.update_layout(height=520, legend_title_text="", clickmode="event+select")
            ev = st.plotly_chart(fig, theme=None, on_select="rerun", selection_mode="points", key="dna_map",
                                 config={"displaylogo": False}, **_CHART_KW)
            sc_ = None
            try:
                pts = ev.selection.points if hasattr(ev, "selection") else ev["selection"]["points"]
                cd = (pts[0].get("customdata") or []) if pts else []
                sc_ = next((int(c_) for c_ in cd if str(c_).isdigit() and int(c_) in p.index), None)
            except Exception:
                sc_ = None
            if sc_ is not None:
                r_ = p.loc[sc_]
                md(ui.action_card(r_["Description"], r_["DNA"],
                                  f"{r_['Category']} · {num(r_['Total_Qty'])} adet · {num(r_['Customers'])} müşteri"
                                  f"<br>Öneri: <b>{r_['Action']}</b>", "info"))
                if st.button("Bu ürün için müşteri bul", type="primary", key="map_go"):
                    goto("Sıradaki 10 Arama", selected_product=int(sc_))
            md(ui.insight(map_note + " <b>Bir noktaya tıklayarak</b> ürünü seçin ve doğrudan müşteri listesine geçin."))
        else:
            chart(st, fig, 520, map_note)
        s = summ.reset_index()
        fig = px.bar(s, x="DNA", y=["Product_Share_%", "Revenue_Share_%"], barmode="group",
                     title="DNA gruplarının ürün sayısı ve ciro içindeki payı (%)",
                     labels={"value": "Pay (%)", "DNA": ""}, color_discrete_sequence=[P["ice"], P["indigo"]])
        fig.for_each_trace(lambda tr_: tr_.update(name=ui.COLS.get(tr_.name, tr_.name)))
        chart(st, fig, 340, f"Ürünlerin yalnızca {ui.pct(summ.loc[STAR, 'Product_Share_%'], 1)} Çok Satan – Düzenli Talep "
                            f"grubunda ama bu grup cironun <b>{ui.pct(summ.loc[STAR, 'Revenue_Share_%'], 1, 'poss_acc')}</b> getiriyor.")

    with t3:
        query = keep_text(st, "Ürün ara (ad veya barkod)", "pf_query", "Otuz Milyon")
        hits = bp.find_products(query, n=30) if query else p.head(0)
        if hits.empty:
            st.info("Bu aramayla eşleşen ürün bulunamadı.")
        else:
            sc = st.selectbox("Ürün", hits.index, format_func=lambda s_: f"{p.loc[s_, 'Description']} ({s_})")
            row = p.loc[sc]
            md(ui.kpi_cards([
                {"label": "DNA grubu", "value": row["DNA"], "icon": "layers", "tone": "indigo"},
                {"label": "Performans puanı", "value": f"{row['Performance_Score']:.0f}/100", "icon": "award", "tone": "indigo"},
                {"label": "Ciro", "value": tl(row["Revenue"]), "icon": "card", "tone": "indigo"},
                {"label": "Müşteri sayısı", "value": num(row["Customers"]), "icon": "users", "tone": "indigo"},
                {"label": "Son 3 ay trendi", "value": spct(100 * row['Trend']),
                 "icon": "trend_up" if row["Trend"] >= 0 else "trend_down", "tone": "indigo" if row["Trend"] >= 0 else "red"},
            ]))
            md(ui.insight(f"<b>Öneri: {row['Action']}</b> — {row['Action_Reason']}",
                          tag="Neden?", risk=row["Action"] in ("İzle (Düşüşte)", "Ürün Performansını İncele")))
            m = bp.df[bp.df["StockCode"] == sc].groupby("Month")["Quantity"].sum()
            m = m.reindex(pd.period_range(bp.df["Month"].min(), bp.df["Month"].max(), freq="M"), fill_value=0)
            fig = px.bar(x=[ui.month_tr(i) for i in m.index], y=m.values, title="Aylık satış adedi",
                         labels={"x": "Ay", "y": "Adet"})
            fig.update_traces(marker_color=ui.DNA_COLORS.get(row["DNA"], P["indigo"]),
                              hovertemplate="%{x}: %{y:,.0f} adet<extra></extra>")
            peak = ui.MONTHS_TR_LONG[int(str(row["Peak_Month"])[5:7]) - 1]
            chart(st, fig, 320, f"En yüksek satış <b>{peak}</b> ayında; ürün yıl boyunca {int(row['Active_Months'])} ayda "
                                f"satıldı ve {num(row['Customers'])} müşteriye ulaştı.")
            if st.button("Bu ürün için müşteri bul (Sıradaki 10 Arama)", type="primary"):
                goto("Sıradaki 10 Arama", selected_product=int(sc))

    with t4:
        cc = portfolio.category_concentration(p, top=40)
        fig = px.scatter(cc.reset_index(), x="Products", y="Top10%_Product_Rev_Share", size="Revenue",
                         color="Limited_History_%", hover_name="Category", log_x=True, color_continuous_scale=ui.SEQ_BLUE,
                         title="Kategoriler: ürün sayısı ve ciro yoğunlaşması",
                         labels={"Products": "Ürün sayısı (log)", "Top10%_Product_Rev_Share": "İlk %10 ürünün ciro payı (%)",
                                 "Limited_History_%": "Yetersiz Satış Geçmişi %"})
        chart(st, fig, 440, "Yukarıdaki kategorilerde ciro birkaç üründe toplanmış; bu ürünlerin bulunurluğu kategori "
                            "cirosu için kritiktir. Balon büyüklüğü kategori cirosunu gösterir.")
        table(cc.round(1), "pf_cat_tbl", hide_index=False, filename="kategori_yogunlasma")


# ===================================================================================
def page_calls(bp: BookPulse):
    p = bp.portfolio
    md(ui.page_header("Sıradaki 10 Arama",
                      "Ürün → müşteri → müşterinin Portföy DNA'sı → sonraki ürün: satış ekibinin kimi, hangi ürün için ve "
                      "neden araması gerektiğini önceliklendirir.", eyebrow="Karar Destek"))
    md(ref_chip())
    t1, t2, t3 = st.tabs(["Fırsat çalışma alanı", "Toplu arama listesi (müşteri bazında)", "Doğrulama"])
    score_cfg = {"Puan": st.column_config.ProgressColumn("Puan", min_value=0, max_value=100, format="%.0f"),
                 "Gerekçe": st.column_config.TextColumn("Gerekçe", width="large")}

    with t1:
        cust_sel = st.session_state.get("selected_customer_calls")
        step_ph = st.empty()
        step_labels = ["Ürün seç", "Müşteriler", "Müşterinin DNA'sı", "Sonraki ürün"]
        step_ph.markdown(ui.steps(step_labels, 1), unsafe_allow_html=True)

        pre = st.session_state.get("selected_product")
        c0, c1, c2, c3 = st.columns([1.6, 1.2, 2.4, 0.9])
        search = c0.text_input("Tüm ürünlerde ara", value="", placeholder="ad veya barkod") \
            if "placeholder" in inspect.signature(st.text_input).parameters else c0.text_input("Tüm ürünlerde ara")
        if search:
            hits = bp.find_products(search, n=50)
            options = list(hits.index)
            dna_sel = None
            c1.caption(f"{len(options)} ürün bulundu")
        else:
            dna_default = p.loc[pre, "DNA"] if pre is not None and pre in p.index else STAR
            st.session_state.setdefault("_calls_dna", dna_default)
            if st.session_state.pop("_calls_dna_sync", False) and pre is not None and pre in p.index:
                st.session_state["_calls_dna"] = p.loc[pre, "DNA"]
            dna_sel = keep_select(c1, "DNA grubu", portfolio.DNA_ORDER, "calls_dna")
            pool = p[p["DNA"] == dna_sel].sort_values("Revenue", ascending=False)
            options = list(pool.index[:1000])
            if pre is not None and pre in pool.index and pre not in options:
                options = [pre] + options
        if not options:
            st.info("Ürün bulunamadı; aramayı değiştirin.")
            return
        sc = c2.selectbox("Ürün", options, index=options.index(pre) if pre in options else 0,
                          format_func=lambda s_: f"{p.loc[s_, 'Description']} · {p.loc[s_, 'DNA']}")
        n = c3.selectbox("Önerilecek müşteri sayısı", [10, 20, 30, 50], index=0)
        if st.session_state.get("selected_product") != int(sc):
            st.session_state["selected_customer_calls"] = None
        st.session_state["selected_product"] = int(sc)

        methods = ["Gelişmiş (önerilen)", "Ekibin ilk modeli", "Özel ağırlıklar"]
        method = st.radio("Puanlama yöntemi", methods,
                          index=methods.index(st.session_state.get("_calls_method", methods[0])), horizontal=True,
                          help="Öncelik etiketi puandan bağımsızdır: Hemen Ara = benzer ürünleri yoğun alan aktif müşteri. Gelişmiş: 6 sinyal, geriye dönük testte en iyi sonuç. Ekibin ilk modeli: kategori miktarı %50, "
                               "sipariş sayısı %30, kategoride son alım %20. Özel: ağırlıkları siz belirleyin.")
        st.session_state["_calls_method"] = method
        weights = None
        if method.startswith("Ekibin"):
            weights = next_calls.V1_WEIGHTS
        elif method.startswith("Özel"):
            with st.expander("Ağırlıklar (toplamı otomatik olarak 100'e ölçeklenir)", expanded=True):
                cw = st.columns(3)
                labels_w = {"category_qty": "Kategori hacmi", "category_share": "Kategori payı",
                            "frequency": "Sipariş sıklığı", "diversity": "Ürün çeşitliliği",
                            "similar_items": "Benzer ürün alımı", "recency": "Yakınlık"}
                weights = {k_: keep_slider(cw[i % 3], lbl, f"w_{k_}", int(100 * config.NEXT_CALLS_WEIGHTS[k_]),
                                           min_value=0, max_value=100, step=5)
                           for i, (k_, lbl) in enumerate(labels_w.items())}
                if sum(weights.values()) == 0:
                    weights = None

        with st.spinner("Müşteriler puanlanıyor…"):
            rec = bp.calls.recommend(sc, n=n, weights=weights)
        if rec.empty:
            st.warning("Bu ürün için aday müşteri bulunamadı (ürünü kategoride herkes almış olabilir).")
            return
        rec = rec.assign(_o=rec["Priority"].map(PRIORITY_ORDER)).sort_values(["_o", "Score"], ascending=[True, False]) \
                 .drop(columns="_o").reset_index(drop=True)

        md(ui.section(f"Öncelikli müşteriler · {p.loc[sc, 'Description']}",
                      "öncelik sırası: Hemen Ara → Geri Kazanım → Teklif Sun → Takip Et; satıra tıklayarak müşteri seçin"))
        pr = rec["Priority"].value_counts()
        md(ui.kpi_cards([
            {"label": "Hemen Ara", "value": int(pr.get("Hemen Ara", 0)), "icon": "phone", "tone": "indigo",
             "note": "güçlü kanıt: benzer ürünleri yoğun alıyor"},
            {"label": "Geri Kazanım", "value": int(pr.get("Geri Kazanım Araması", 0)), "icon": "refresh", "tone": "red",
             "note": f"{config.DORMANT_DAYS}+ gündür alışveriş yok"},
            {"label": "Teklif Sun / Takip", "value": int(pr.get("Teklif Sun", 0) + pr.get("Takip Et", 0)),
             "icon": "briefcase", "tone": "indigo", "note": "orta veya zayıf benzer ürün kanıtı"},
            {"label": "Ürün DNA'sı", "value": p.loc[sc, "DNA"], "icon": "layers", "tone": "indigo",
             "note": f"ürünü şimdiye dek {num(p.loc[sc, 'Customers'])} müşteri aldı"},
        ]))
        view = rec[["CustomerID", "Priority", "Score", "Reason", "Category_Qty", "Category_Orders", "Similar_Score",
                    "Days_Since_Last", "City"]]
        kw = {**_TABLE_KW, "hide_index": True,
              "column_config": {**score_cfg, **{c_: st.column_config.Column(help=ui.term_help(c_))
                                                for c_ in ui.tr_df(view).columns if ui.term_help(c_)}}}
        picked = None
        if _ROW_SELECT:
            ev = st.dataframe(ui.tr_df(view), on_select="rerun", selection_mode="single-row", key="calls_tbl", **kw)
            try:
                rows = ev.selection.rows if hasattr(ev, "selection") else ev["selection"]["rows"]
                if rows:
                    picked = rec.loc[rows[0], "CustomerID"]
            except Exception:
                picked = None
        else:
            st.dataframe(ui.tr_df(view), **kw)
        ids = list(rec["CustomerID"])
        default_c = picked or (cust_sel if cust_sel in ids else ids[0])
        cust = st.selectbox("Seçili müşteri", ids, index=ids.index(default_c),
                            format_func=lambda c_: f"{c_} · {rec.set_index('CustomerID').loc[c_, 'Priority']} · puan "
                                                   f"{rec.set_index('CustomerID').loc[c_, 'Score']:.0f}")
        st.session_state["selected_customer_calls"] = cust
        step_ph.markdown(ui.steps(step_labels, 2 if (picked is None and cust == ids[0]) else 3), unsafe_allow_html=True)
        st.download_button("Arama listesini indir (CSV)", ui.tr_df(rec).to_csv(index=False).encode("utf-8-sig"),
                           f"siradaki10_{sc}.csv", "text/csv")

        md(ui.section(f"Müşteri: {cust}", "neden bu müşteri ve satın alma portföyü"))
        why = bp.calls.why_customer(cust, sc)
        row = rec.set_index("CustomerID").loc[cust]
        sims_txt = ", ".join(why["similar_products_bought"][:4]) or "–"
        l, r = st.columns([1, 1.25], gap="medium")
        md(ui.action_card("Neden bu müşteri?", f"{row['Priority']} · {row['Score']:.0f}/100",
                          f"{row['Reason']}<br><br>• '{why['category']}' kategorisi alımlarının "
                          f"{ui.pct(why['category_share_%'], 1)}<br>• Bu ürünle birlikte alınan ürünlerden "
                          f"<b>{why['n_similar_products_bought']}</b> tanesini almış: {sims_txt}<br>"
                          f"• Son alışveriş {why['days_since_last']} gün önce (referans: 31.12.2020)",
                          "risk" if row["Priority"] == "Geri Kazanım Araması" else "info"), l)
        cp = bp.calls.customer_portfolio(cust)
        if not cp.empty:
            comp = cp.reset_index()[["DNA", "Customer_Share_%", "Company_Share_%"]].melt(
                "DNA", var_name="Kapsam", value_name="Pay")
            comp["Kapsam"] = comp["Kapsam"].map({"Customer_Share_%": cust, "Company_Share_%": "Şirket geneli"})
            fig = px.bar(comp, x="DNA", y="Pay", color="Kapsam", barmode="group",
                         title="Alımların Portföy DNA dağılımı (adet %)", labels={"Pay": "Pay (%)", "DNA": ""},
                         color_discrete_sequence=[P["indigo"], P["ice"]])
            big = cp["Difference_pp"].idxmax()
            chart(r, fig, 330, f"Bu müşteri <b>{big}</b> grubundan şirket ortalamasına göre "
                               f"{cp.loc[big, 'Difference_pp']:+.1f} puan daha fazla alıyor.")

        md(ui.section("Sonraki ürün", "müşterinin henüz almadığı, benzer müşterilerin aldığı ürünler"))
        recs = bp.calls.products_for_customer(cust, n=12, dna_filter=None)
        if recs.empty:
            st.info("Bu müşteri için yeni ürün önerisi bulunamadı.")
        else:
            c1, c2 = st.columns([3, 1])
            new_sc = c1.selectbox("Önerilen ürün", list(recs["StockCode"]),
                                  format_func=lambda s_: f"{bp.products.loc[s_, 'Description']} · "
                                                         f"{p.loc[s_, 'DNA'] if s_ in p.index else ''}")
            c2.write("")
            if c2.button("Bu ürün için yeni liste", type="primary"):
                goto("Sıradaki 10 Arama", selected_product=int(new_sc), selected_customer_calls=None)
            show(st, recs.drop(columns=["StockCode"]), hide_index=True)
        b1, b2 = st.columns(2)
        if b1.button("Müşterinin tam profilini aç"):
            goto("Müşteri DNA", selected_customer=cust, cust_tab="Müşteri profili")
        if b2.button("Bu müşteriyi Yapay Zekâ Danışmanına sor"):
            goto("Yapay Zekâ Satış Danışmanı", last_page="Sıradaki 10 Arama")

    with t2:
        md(ui.insight("Puanlar her ürün için müşteriler arasındaki <b>göreli sıradır</b>; farklı ürünlerin puanları "
                      "birbiriyle karşılaştırılmamalı. Bu yüzden liste <b>müşteri bazında</b> birleştirilir: her müşteri "
                      "için tek kart, birden çok ürün fırsatı. Sıralama: fırsat sayısı, sonra müşterinin yıllık cirosu.",
                      tag="Nasıl okunur?"))
        c1, c2, c3 = st.columns(3)
        dnas = keep_multi(c1, "DNA grupları", [STAR, VOLATILE, "Az Satan – Düzenli Talep"], "batch_dna") or [STAR, VOLATILE]
        n_products = c2.selectbox("Ürün sayısı (ciroya göre ilk N)", [50, 100, 200, 500], index=1)
        per = c3.selectbox("Ürün başına müşteri", [5, 10, 20], index=1)
        b = run_batch(tuple(dnas), int(n_products), int(per))
        if b.empty:
            st.info("Liste boş.")
        else:
            opp = bp.calls.customer_opportunities(b)
            md(ui.kpi_cards([
                {"label": "Aranacak müşteri", "value": num(len(opp)), "icon": "users", "tone": "indigo"},
                {"label": "Toplam ürün fırsatı", "value": num(len(b)), "icon": "list", "tone": "indigo"},
                {"label": "Geri kazanım önceliği", "value": num((opp["Priority"] == "Geri Kazanım Araması").sum()),
                 "icon": "refresh", "tone": "red", "note": "uzun süredir alışveriş yapmayan"},
            ]))
            table(opp, "batch_opp", hide_index=False, filename="toplu_arama_musteri_bazinda")
            with st.expander("Ürün bazında ayrıntı (tüm satırlar)"):
                table(b.drop(columns=["StockCode"]), "batch_rows", filename="toplu_arama_detay")

    with t3:
        bt = stored_backtest()
        md(ui.insight("Model <b>Ocak–Eylül</b> verisiyle kurulur; önerilen müşterinin ürünü <b>Ekim–Aralık</b>'ta gerçekten "
                      "alıp almadığı ölçülür. Ağırlıklar ayrı bir doğrulama penceresinde (Ocak–Haziran → Temmuz–Eylül) "
                      "seçildi. Sonuç önceden hesaplanmıştır (<code>tools/precompute_backtest.py</code>).",
                      tag="Yöntem"))
        if st.button("Testi yeniden çalıştır (≈30 sn)"):
            bt = run_backtest(300)
        if bt:
            t = bt["table"].reset_index().rename(columns={"index": "Model"})
            fig = px.bar(t, x="Precision@10_%", y="Model", orientation="h", text="Precision@10_%",
                         title=f"İsabet@10: önerilen 10 müşteriden kaçı ürünü gerçekten aldı? ({bt['n_products']} ürün)",
                         labels={"Precision@10_%": "İsabet@10 (%)", "Model": ""})
            fig.update_traces(marker_color=[P["indigo"], P["sky"], P["gray"]], texttemplate="%%{text:.1f}", cliponaxis=False,
                              textposition="outside")
            fig.update_yaxes(autorange="reversed")
            v2, v1, rnd = t["Precision@10_%"].tolist()[:3]
            chart(st, fig, 280, f"Gelişmiş model önerdiği müşterilerin <b>{ui.pct(v2, 1, 'poss_acc')}</b> doğru buldu; "
                                f"ekibin ilk modeli {pct1(v1)}, rastgele seçim {pct1(rnd)}. Rastgeleye göre yaklaşık "
                                f"<b>{v2 / max(rnd, 0.01):.0f} kat</b> daha isabetli.")
            show(st, bt["table"])
            if bt.get("priority"):
                pt = pd.DataFrame(bt["priority"]).set_index("Öncelik")
                pt = pt.reindex([k_ for k_ in PRIORITY_ORDER if k_ in pt.index])
                md(ui.section("Öncelik etiketleri işe yarıyor mu?", "Ekim–Aralık'ta ürünü gerçekten alma oranı"))
                fig = px.bar(pt.reset_index(), x="Öncelik", y="İsabet %", text="İsabet %",
                             title="Öncelik etiketine göre isabet (%)", labels={"Öncelik": "", "İsabet %": "İsabet (%)"})
                fig.update_traces(marker_color=[P["indigo"] if k_ == "Hemen Ara" else P["coral"] if k_.startswith("Geri")
                                                else P["sky"] if k_ == "Teklif Sun" else P["gray"] for k_ in pt.index],
                                  texttemplate="%%{text:.1f}", textposition="outside", cliponaxis=False)
                chart(st, fig, 300, "Etiketler göreli puandan değil, <b>benzer ürün kanıtından</b> gelir: "
                                    "<i>Hemen Ara</i> listesindeki müşterilerin ürünü alma oranı <i>Takip Et</i>'tekilerden "
                                    "belirgin biçimde yüksek.")
                show(st, pt)


# ===================================================================================
def page_customers(bp: BookPulse):
    c = bp.customers
    summ = customer_dna.segment_summary(c)
    md(ui.page_header("Müşteri DNA", "Bir müşterinin tüm resmi tek ekranda: değeri, alışkanlıkları, ürün portföyü ve "
                                     "sonraki satış fırsatı. Segment ve küme analizleri ayrıntı sekmelerindedir.",
                      eyebrow="Müşteri Analitiği"))
    tabs = ["Müşteri profili", "Kaçan Sipariş Dedektörü", "Segmentler", "Kümeler", "Segment × Portföy DNA"]
    wanted = st.session_state.pop("cust_tab", None)
    if wanted and wanted != tabs[0]:
        md(ui.insight(f"İstediğiniz analiz <b>{wanted}</b> sekmesinde.", tag="Yönlendirme"))
    t1, t2, t3, t4, t5 = st.tabs(tabs)

    with t1:
        ids = list(c.sort_values("Monetary", ascending=False).index)
        cid = keep_select(st, "Müşteri (ciroya göre sıralı)", ids, "cust_id")
        st.session_state["selected_customer"] = cid
        row = c.loc[cid]
        risky = row["Segment"] in ("Riskli Müşteriler", "Riskli Değerli Müşteriler", "Pasif Müşteriler")
        md(ui.kpi_cards([
            {"label": "Segment", "value": row["Segment"], "icon": "layers", "tone": "red" if risky else "indigo"},
            {"label": "Ciro", "value": tl(row["Monetary"]), "icon": "card", "tone": "indigo",
             "note": f"müşteriler arasında {ids.index(cid) + 1}. sırada"},
            {"label": "Sipariş", "value": num(row["Frequency"]), "icon": "receipt", "tone": "indigo",
             "note": f"ort. {tl(row['Avg_Order_Value'])}"},
            {"label": "Son alım", "value": f"{int(row['Recency'])} gün önce", "icon": "clock",
             "tone": "red" if row["Recency"] > config.DORMANT_DAYS else "indigo", "note": "referans 31.12.2020"},
            {"label": "Kategori çeşitliliği", "value": num(row["Categories"]), "icon": "book", "tone": "indigo",
             "note": f"küme: {row['Cluster_Name']}"},
        ]))
        md(ui.insight(f"<b>Önerilen yaklaşım:</b> {row['Action']}", tag="Ne yapmalı?", risk=risky))
        d = bp.df[bp.df["CustomerID"] == cid]
        l, r = st.columns(2, gap="medium")
        m = d.groupby("Month")["Revenue"].sum()
        m = m.reindex(pd.period_range(bp.df["Month"].min(), bp.df["Month"].max(), freq="M"), fill_value=0)
        fig = px.bar(x=[ui.month_tr(i) for i in m.index], y=m.values, title="Aylık alım (₺)", labels={"x": "Ay", "y": "Ciro (₺)"})
        fig.update_traces(marker_color=P["indigo"], hovertemplate="%{x}: %{y:,.0f} ₺<extra></extra>")
        active = int((m > 0).sum())
        chart(l, fig, 320, f"Müşteri 12 ayın <b>{active}</b> ayında alım yaptı; en yüksek ay "
                           f"{ui.MONTHS_TR_LONG[int(np.argmax(m.values))]}.")
        cat = d.groupby("Category")["Revenue"].sum().nlargest(8).sort_values()
        fig = px.bar(x=cat.values, y=cat.index, orientation="h", title="En çok alınan 8 kategori (₺)",
                     labels={"x": "Ciro (₺)", "y": ""})
        fig.update_traces(marker_color=P["sky"], hovertemplate="%{y}: %{x:,.0f} ₺<extra></extra>")
        chart(r, fig, 320, f"Cironun {ui.pct(100 * cat.iloc[-1] / max(d['Revenue'].sum(), 1))} <b>{cat.index[-1]}</b> "
                           "kategorisinden." if len(cat) else None)
        cp = bp.calls.customer_portfolio(cid)
        if not cp.empty:
            comp = cp.reset_index()[["DNA", "Customer_Share_%", "Company_Share_%"]].melt(
                "DNA", var_name="Kapsam", value_name="Pay")
            comp["Kapsam"] = comp["Kapsam"].map({"Customer_Share_%": cid, "Company_Share_%": "Şirket geneli"})
            fig = px.bar(comp, x="DNA", y="Pay", color="Kapsam", barmode="group",
                         title="Müşterinin Portföy DNA'sı ve şirket geneli (adet %)", labels={"Pay": "Pay (%)", "DNA": ""},
                         color_discrete_sequence=[P["indigo"], P["ice"]])
            big = cp["Difference_pp"].idxmax()
            chart(st, fig, 300, f"Müşteri <b>{big}</b> ürünlerine şirket ortalamasından {cp.loc[big, 'Difference_pp']:+.1f} "
                                "puan daha fazla yöneliyor.")
        md(ui.section("Sonraki satış fırsatı", "benzer müşterilerin aldığı, bu müşterinin henüz almadığı ürünler"))
        recs = bp.calls.products_for_customer(cid, n=10)
        if recs.empty:
            st.info("Bu müşteri için öneri bulunamadı.")
        else:
            show(st, recs.drop(columns=["StockCode"]), hide_index=True)
            c1, c2 = st.columns([3, 1])
            sel = c1.selectbox("Ürün seç", list(recs["StockCode"]), format_func=lambda s_: bp.products.loc[s_, "Description"],
                               key="cust_prod_pick")
            c2.write("")
            if c2.button("Bu ürün için Sıradaki 10 Arama"):
                goto("Sıradaki 10 Arama", selected_product=int(sel), selected_customer_calls=None)

    with t2:
        mo = missing_orders(bp)
        md(ui.kpi_cards([
            {"label": "İşaretlenen müşteri", "value": num(len(mo)), "icon": "alert", "tone": "red",
             "note": "tipik sipariş aralığının ≥2 katı ve ≥21 gün sessiz"},
            {"label": "Bu müşterilerin yıllık cirosu", "value": tl_m(mo["Revenue"].sum()), "icon": "card", "tone": "red",
             "note": f"toplam cironun {ui.pct(100 * mo['Revenue'].sum() / bp.kpis['revenue'], 1)}"},
            {"label": "Medyan gecikme", "value": f"{mo['Delay_Ratio'].median():.1f} kat".replace(".", ","), "icon": "clock",
             "tone": "indigo", "note": "tipik sipariş aralığına göre"},
        ]))
        top = mo.head(25).reset_index()
        fig = px.scatter(top, x="Days_Since_Last", y="Revenue", size="Order_Days", color="Delay_Ratio",
                         hover_name="CustomerID", color_continuous_scale=ui.SEQ_RED, log_y=True,
                         title="Ciroya göre ilk 25 işaretli müşteri: sessizlik süresi ve yıllık ciro",
                         labels={"Days_Since_Last": "Son siparişten beri (gün)", "Revenue": "Yıllık ciro (₺, log)",
                                 "Order_Days": "Sipariş günü", "Delay_Ratio": "Gecikme (kat)"})
        lead = mo.index[0] if len(mo) else "-"
        chart(st, fig, 400, f"Sağ üstteki müşteriler hem büyük hem uzun süredir sessiz. Örneğin <b>{lead}</b> normalde "
                            f"{mo.iloc[0]['Typical_Gap']:.0f} günde bir sipariş verirken {mo.iloc[0]['Days_Since_Last']:.0f} "
                            "gündür sipariş vermedi. Geriye dönük kontrolde işaretlenenlerin %59'u sonraki 3 ayda dönmedi."
              if len(mo) else None, risk=True)
        table(mo, "mo_tbl", hide_index=False, filename="kacan_siparisler")
        if len(mo):
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("Müşteri profiline git", list(mo.index[:50]), key="mo_pick")
            c2.write("")
            if c2.button("Profili aç"):
                goto("Müşteri DNA", selected_customer=pick)

    with t3:
        l, r = st.columns(2, gap="medium")
        s = summ.reset_index()
        fig = px.bar(s, x="Segment", y=["Customer_Share_%", "Revenue_Share_%"], barmode="group",
                     title="Segmentlerin müşteri ve ciro payı (%)", labels={"value": "Pay (%)", "Segment": ""},
                     color_discrete_sequence=[P["ice"], P["indigo"]])
        fig.for_each_trace(lambda tr_: tr_.update(name=ui.COLS.get(tr_.name, tr_.name)))
        fig.update_xaxes(tickangle=-25)
        champ = summ.loc["En Değerli Müşteriler"] if "En Değerli Müşteriler" in summ.index else None
        chart(l, fig, 400, (f"En Değerli Müşteriler segmenti müşterilerin {ui.pct(champ['Customer_Share_%'], 1)} ama cironun "
                            f"<b>{ui.pct(champ['Revenue_Share_%'], 1)}</b>.") if champ is not None else None)
        fig = px.scatter(c.reset_index(), x="Recency", y="Frequency", size="Monetary", color="Segment",
                         color_discrete_map=ui.SEG_COLORS, log_y=True, hover_name="CustomerID", size_max=30,
                         title="Yakınlık ve sipariş sıklığı (balon = ciro)", opacity=0.75,
                         labels={"Recency": "Son alımdan beri (gün)", "Frequency": "Sipariş sayısı (log)"},
                         category_orders={"Segment": customer_dna.SEGMENT_ORDER})
        chart(r, fig, 400, "Sol üst: sık ve yakın zamanda alanlar (değerli). Sağ: uzun süredir almayanlar (risk).")
        table(summ, "seg_summ", hide_index=False, filename="segment_ozeti")
        seg = keep_select(st, "Segmentteki müşteriler", list(summ.index), "seg_pick")
        table(c[c["Segment"] == seg].sort_values("Monetary", ascending=False)[
            ["City", "CustomerType", "Recency", "Frequency", "Monetary", "Categories", "RFM_Score", "Cluster_Name"]],
            "seg_list", hide_index=False, filename=f"segment_{seg}")

    with t4:
        info = bp.cluster_info
        sil = pd.Series(info["silhouette"]).rename_axis("k").reset_index(name="silhouette")
        l, r = st.columns([1, 2], gap="medium")
        fig = px.line(sil, x="k", y="silhouette", markers=True, title="Küme sayısına göre silhouette",
                      labels={"k": "Küme sayısı (k)", "silhouette": "Silhouette"})
        fig.update_traces(line_color=P["indigo"], marker=dict(size=9))
        chart(l, fig, 320, f"En net ayrışma <b>k = {info['best_k']}</b> kümede.")
        show(r, info["profile"], height=320)
        fig = px.scatter(c.reset_index(), x="Recency", y="Monetary", color="Cluster_Name", log_y=True,
                         hover_name="CustomerID", opacity=0.7, title="Kümeler: son alım ve ciro",
                         labels={"Recency": "Son alımdan beri (gün)", "Monetary": "Ciro (₺, log)", "Cluster_Name": "Küme"},
                         color_discrete_sequence=[P["indigo"], P["sky"], P["coral"], P["amber"], P["gray"]])
        chart(st, fig, 420, "Kümeler müşterileri değer ve aktifliğe göre üç gruba ayırıyor; segment tablosuyla çapraz "
                            "kontrol aşağıda.")
        show(st, pd.crosstab(c["Segment"], c["Cluster_Name"]))

    with t5:
        x = customer_dna.segment_x_portfolio(bp.df, c, bp.portfolio)
        fig = px.imshow(x, text_auto=".0f", aspect="auto", color_continuous_scale=ui.SEQ_BLUE,
                        labels=dict(x="Portföy DNA", y="Segment", color="%"),
                        title="Segment cirosunun Portföy DNA gruplarına dağılımı (satır %)")
        ra = x.loc["Riskli Müşteriler", VOLATILE] if "Riskli Müşteriler" in x.index else None
        chart(st, fig, 440, (f"Riskli Müşteriler segmentinin cirosunun <b>{ui.pct(ra, 0, 'poss')}</b> Çok Satan – Değişken Talep "
                             "grubundaki ürünlerden geliyor: "
                             "büyük olasılıkla sezonluk alıcılar; geri kazanım sezon öncesine planlanmalı.")
              if ra is not None else None)


# ===================================================================================
def page_basket(bp: BookPulse):
    rules = bp.product_rules
    md(ui.page_header("Sepet Analizi", "Hangi ürünler aynı faturada birlikte satın alınıyor? Çapraz satış ve paket kampanyası "
                                       "fırsatları.", eyebrow="Satış ve Talep",
                      chip=f"{num(rules.attrs.get('n_baskets', 0))} sepet"))
    top_b = basket.bundles(rules, n=6, min_confidence=0.3, cross_category_only=False)
    md(ui.section("En güçlü birlikte satış çiftleri", "paket kampanyası adayları"))
    cols = st.columns(3)
    for i, (_, r_) in enumerate(top_b.iterrows()):
        md(ui.action_card(f"Paket {i + 1}", f"%{100 * r_['Confidence']:.0f} birlikte",
                          f"<b>{r_['A_Name']}</b> alanların %{100 * r_['Confidence']:.0f}'si <b>{r_['B_Name']}</b> de "
                          f"alıyor ({num(r_['Count_AB'])} sepet).", "info"), cols[i % 3])
    md(ui.insight("Çiftlerin çoğu aynı serinin kitapları: seri setleri hazır paket olarak sunulabilir. Farklı kategorilerden "
                  "fırsatlar için <i>Paket önerileri</i> sekmesindeki filtreyi açın.", tag="Ne yapmalı?"))
    t1, t2, t3 = st.tabs(["Çapraz satış", "Paket önerileri", "Kategori birliktelikleri"])

    with t1:
        query = keep_text(st, "Ürün ara", "bk_query", "Şeker Portakalı")
        hits = bp.find_products(query, n=30) if query else bp.portfolio.head(0)
        hits = hits[hits.index.isin(rules["A"].unique())]
        if hits.empty:
            st.info("Bu aramayla eşleşen ve birliktelik kuralı olan ürün yok. Daha çok satan bir ürün deneyin.")
        else:
            sc = st.selectbox("Ürün", hits.index, format_func=lambda s_: bp.products.loc[s_, "Description"])
            cs = basket.cross_sell(rules, sc, n=12)
            fig = px.bar(cs.sort_values("Confidence"), x="Confidence", y="B_Name", orientation="h",
                         title="Bu ürünü alanlara önerilecek ürünler",
                         labels={"Confidence": "Birlikte alınma oranı (güven)", "B_Name": ""},
                         hover_data={"Lift": ":.1f", "Count_AB": True})
            fig.update_traces(marker_color=P["indigo"])
            fig.update_layout(xaxis_tickformat=".0%")
            best = cs.iloc[0] if len(cs) else None
            chart(st, fig, 420, (f"Bu ürünü alan faturaların <b>%{100 * best['Confidence']:.0f}</b>'inde "
                                 f"<b>{best['B_Name']}</b> de var (tesadüfe göre {best['Lift']:.1f} kat daha sık).")
                  if best is not None else None)
            table(cs[["B_Name", "B_Category", "Count_AB", "Support", "Confidence", "Lift"]].round(4), "bk_cs",
                  filename="capraz_satis")

    with t2:
        c1, c2 = st.columns(2)
        cross = getattr(c1, "toggle", c1.checkbox)("Yalnızca farklı kategorilerden eşleşmeler", value=False,
                                                   help="Açıkken aynı serinin ciltleri yerine farklı kategorilerden çiftler.")
        min_conf = c2.slider("En düşük güven", 0.05, 0.9, 0.2, 0.05, help=ui.GLOSSARY_EXACT["Güven"])
        b = basket.bundles(rules, n=200, min_confidence=min_conf, cross_category_only=cross)
        table(b[["A_Name", "B_Name", "A_Category", "B_Category", "Count_AB", "Confidence", "Lift"]].round(3), "bk_bundles",
              filename="paket_onerileri")
        if len(b):
            md(ui.section("Kampanya paketi oluştur", "seçtiğiniz çiftlerden indirilebilir paket listesi"))
            labels_b = [f"{r_['A_Name']} + {r_['B_Name']}" for _, r_ in b.head(50).iterrows()]
            chosen = keep_multi(st, "Pakete eklenecek çiftler (ilk 50 öneri arasından)", labels_b, "bk_pack")
            if chosen:
                pack = b.head(50).iloc[[labels_b.index(x) for x in chosen]]
                buyers = pack["Count_AB"].sum()
                md(ui.insight(f"<b>{len(pack)}</b> paket seçildi; bu çiftler geçmişte toplam <b>{num(buyers)}</b> sepette "
                              "birlikte satın alındı. Listeyi kampanya planlaması için indirebilirsiniz.", tag="Paket"))
                st.download_button("Paket listesini indir (CSV)",
                                   ui.tr_df(pack[["A_Name", "B_Name", "Count_AB", "Confidence", "Lift"]])
                                   .to_csv(index=False).encode("utf-8-sig"), "kampanya_paketi.csv", "text/csv")

    with t3:
        cr = bp.category_rules
        top_cats = bp.df.groupby("Category")["Revenue"].sum().nlargest(20).index
        sub = cr[cr["A"].isin(top_cats) & cr["B"].isin(top_cats)]
        mat = sub.pivot_table(index="A", columns="B", values="Lift", aggfunc="max")
        fig = px.imshow(mat, aspect="auto", color_continuous_scale=ui.SEQ_BLUE,
                        labels=dict(x="B kategorisi", y="A kategorisi", color="Kaldıraç"),
                        title="Ciroya göre ilk 20 kategori arasında birliktelik gücü (kaldıraç, A → B)")
        chart(st, fig, 620, "Koyu hücreler birlikte alınma eğiliminin tesadüften güçlü olduğu kategori çiftleridir; "
                            "boş hücrelerde anlamlı birliktelik yok.")
        table(cr[["A", "B", "Count_AB", "Support", "Confidence", "Lift"]].round(3), "bk_cat", filename="kategori_birliktelik")

    with st.expander("Yöntem notu"):
        sizes = bp.df.groupby("InvoiceID")["StockCode"].nunique()
        big = sizes > 150
        md(ui.insight(f"150 kalemden büyük <b>{num(big.sum())}</b> fatura (tüm faturaların "
                      f"{ui.pct(100 * big.mean(), 1)}) dışarıda kaldı; bu faturalar satılan adedin "
                      f"{ui.pct(100 * bp.df[bp.df['InvoiceID'].isin(sizes[big].index)]['Quantity'].sum() / bp.df['Quantity'].sum(), 1)} "
                      "içeriyor. Sınır kaldırılırsa neredeyse her ürün çifti birbiriyle ilişkili görünür.", tag="Etki"))
        md(ui.glossify("Sepet = bir fatura. 2'den az veya 150'den fazla kalem içeren faturalar dışarıda bırakılır: B2B'de çok "
                       "büyük faturalar neredeyse her ürünü içerdiği için sahte birliktelikler üretir. Kurallar en az 20 sepette "
                       "görülen çiftlerden, Güven ≥ %5 ve kaldıraç ≥ 1,2 koşuluyla üretilir."))


# ===================================================================================
def page_forecast(bp: BookPulse):
    md(ui.page_header("Talep Tahmini", "Kategori bazında, 2020 sonundan itibaren 8 haftalık talep (adet) tahmini; gerçekleşen "
                                       "değerler ve doğrulama dönemindeki tahminlerle karşılaştırmalı.",
                      eyebrow="Satış ve Talep"))
    md(ref_chip())
    S, ev, fc = bp.weekly, bp.forecast_eval, bp.forecast_table
    gb = ev["global_best"]
    sw, rw = ev["selection_window"], ev["report_window"]
    rep = ev["overall"]
    md(ui.kpi_cards([
        {"label": "Kullanılan yöntem", "value": METHOD_TR.get(gb, gb), "icon": "award", "tone": "indigo",
         "note": f"seçim penceresi {dtr(sw[0])} – {dtr(sw[1])}"},
        {"label": "Test hatası (WAPE)", "value": pct1(rep.loc[gb, "WAPE_%"]), "icon": "activity", "tone": "indigo",
         "note": f"bağımsız test {dtr(rw[0])} – {dtr(rw[1])}"},
        {"label": "Gradient boosting hatası", "value": pct1(rep.loc["GBM", "WAPE_%"]), "icon": "activity", "tone": "indigo",
         "note": "karmaşık model bu veride daha kötü"},
        {"label": "Tahmin ufku", "value": "8 hafta", "icon": "clock", "tone": "indigo", "note": "28.12.2020'den itibaren"},
    ]))
    series = keep_select(st, "Seri (kategori veya toplam)", list(S.columns), "fc_series", default="TOPLAM")
    f = fc[fc["Series"] == series]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f["Week"]) + list(f["Week"])[::-1], y=list(f["Upper"]) + list(f["Lower"])[::-1],
                             fill="toself", fillcolor="rgba(91,95,227,0.12)", line=dict(width=0), name="Belirsizlik bandı", mode="lines",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=S.index, y=S[series], name="Gerçekleşen", mode="lines+markers",
                             line=dict(color=P["indigo"], width=2.4), marker=dict(size=4),
                             hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} adet<extra>Gerçekleşen</extra>"))
    vp = ev["preds"][gb][series]
    fig.add_trace(go.Scatter(x=vp.index, y=vp.values, name="Doğrulama tahmini", mode="lines", line=dict(color=P["amber"], dash="dot"),
                             hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} adet<extra>Doğrulama tahmini</extra>"))
    fig.add_trace(go.Scatter(x=f["Week"], y=f["Forecast"], name="Tahmin", mode="lines", line=dict(color=P["sky"], width=3),
                             hovertemplate="%{x|%d.%m.%Y}: %{y:,.0f} adet<extra>Tahmin</extra>"))
    fig.add_vrect(x0=sw[0], x1=rw[1], fillcolor=P["gray"], opacity=0.12, line_width=0,
                  annotation_text="değerlendirme dönemi", annotation_position="top left")
    fig.update_layout(title=f"{series}: haftalık satış adedi – gerçekleşen, doğrulama ve tahmin",
                      hovermode="x unified", xaxis_title="Hafta", yaxis_title="Adet",
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
    act = ev["actual"][series].loc[pd.Timestamp(rw[0]):]
    prd = ev["preds"][gb][series].loc[pd.Timestamp(rw[0]):]
    err = 100 * (prd - act).abs().sum() / max(act.sum(), 1)
    last4, next4 = S[series].tail(4).sum(), f["Forecast"].head(4).sum()
    chart(st, fig, 460, f"Test haftalarında tahmin, gerçekleşenden ortalama <b>%{err:.0f}</b> saptı. Önümüzdeki 4 hafta için "
                        f"{num(next4)} adet bekleniyor (son 4 hafta: {num(last4)}; {spct(100 * (next4 - last4) / max(last4, 1))})."
                        " Bant istatistiksel bir varsayımdır, garanti edilen bir güven aralığı değildir.")
    comp = pd.DataFrame({"Week": act.index, "Gerçekleşen": act.values, "Tahmin": prd.values})
    comp["Fark %"] = (100 * (comp["Tahmin"] - comp["Gerçekleşen"]) / comp["Gerçekleşen"].replace(0, np.nan)).round(1)
    l, r = st.columns(2, gap="medium")
    with l:
        md(ui.section("Gerçekleşen ve tahmin", "bağımsız test haftaları"))
        show(st, comp.set_index("Week").round(0))
    with r:
        md(ui.section("Önümüzdeki 4 hafta", "kategori bazında"))
        out = forecast.outlook(S, fc, 4)
        out["Yöntem"] = out["Yöntem"].map(lambda m_: METHOD_TR.get(m_, m_))
        show(st, out, column_config={"Değişim %": st.column_config.NumberColumn("Değişim %", format="%+.1f")})
    with st.expander("Yöntem karşılaştırması ve değerlendirme tasarımı"):
        md(ui.glossify(f"Son 8 hafta ayrı tutulur. İlk 4 hafta ({dtr(sw[0])} – {dtr(sw[1])}) <b>yöntem seçimi</b>, son 4 hafta "
                       f"({dtr(rw[0])} – {dtr(rw[1])}) <b>bağımsız test</b> için kullanılır; aynı dönemde hem seçip hem raporlamak "
                       "sonucu iyimser gösterirdi. Tüm seriler için tek yöntem seçilir (seri bazında seçim daha kötü "
                       "genellendi)."))
        show(st, rep.rename(index=lambda i: METHOD_TR.get(i, i)))
        ps = ev["per_series"].rename(columns=lambda c_: METHOD_TR.get(c_, c_))
        ps["Best"] = ps["Best"].map(lambda m_: METHOD_TR.get(m_, m_))
        show(st, ps)
    st.download_button("Tahminleri indir (CSV)", ui.tr_df(fc).to_csv(index=False).encode("utf-8-sig"), "tahminler.csv",
                       "text/csv")


# ===================================================================================
EXAMPLES = [
    "Bu müşteriyi neden aramalıyım?",
    "Bu ürün için başka kimleri aramalıyım?",
    "Son üç ayda satışları düşen kategoriler hangileri?",
    "Sipariş vermeyi geciktiren büyük müşteriler kimler?",
    "Birlikte en çok satın alınan ürünler neler?",
    "Önümüzdeki ay hangi kategoride daha fazla satış bekleniyor?",
    "Çok Satan – Düzenli Talep grubundaki en önemli ürünler hangileri?",
    "Toplam ciro ve müşteri sayısı nedir?",
]
TOOL_TR = {
    "get_kpis": "Genel özet", "top_entities": "En iyiler listesi", "category_trend": "Kategori trendi",
    "monthly_sales": "Aylık satışlar", "portfolio_products": "Portföy ürünleri", "portfolio_summary": "Portföy özeti",
    "search_product": "Ürün arama", "next_calls_for_product": "Ürün için arama listesi", "top_call_list": "Toplu arama listesi",
    "products_for_customer": "Müşteriye ürün önerisi", "customer_profile": "Müşteri profili",
    "customer_segments": "Müşteri segmentleri", "cross_sell": "Çapraz satış", "top_bundles": "Paket önerileri",
    "demand_forecast": "Talep tahmini", "customer_portfolio_dna": "Müşteri Portföy DNA'sı",
    "why_customer": "Neden bu müşteri?", "missing_orders": "Kaçan Sipariş Dedektörü",
}


def _render_tool_result(name, args, out):
    arg_txt = ", ".join(f"{k}={v}" for k, v in (args or {}).items())
    with st.expander(f"Kullanılan analiz: {TOOL_TR.get(name, name)}" + (f" · {arg_txt}" if arg_txt else ""), expanded=False):
        if "error" in out:
            st.error(out["error"])
            return
        for key, val in out.items():
            if isinstance(val, dict) and "columns" in val and "rows" in val:
                t = pd.DataFrame(val["rows"], columns=val["columns"])
                show(st, t, hide_index=True)
                num_cols = [c for c in t.columns if pd.api.types.is_numeric_dtype(t[c]) and c not in ("StockCode",)]
                txt_cols = [c for c in t.columns if not pd.api.types.is_numeric_dtype(t[c])]
                ch = out.get("chart")
                if ch:
                    fig = px.area(t, x=ch["x"], y=ch["y"], markers=True)
                    fig.update_traces(line_color=P["indigo"], fillcolor="rgba(91,95,227,.12)")
                    chart(st, fig, 300)
                elif txt_cols and num_cols and 2 <= len(t) <= 25:
                    fig = px.bar(t.iloc[::-1], x=num_cols[0], y=txt_cols[0], orientation="h")
                    fig.update_traces(marker_color=P["indigo"])
                    chart(st, fig, 340)
            elif not isinstance(val, dict):
                st.write(f"**{ui.COLS.get(key, key)}:** {val}")
            else:
                st.json(val, expanded=False)


@st.cache_data(ttl=30, show_spinner=False)
def _ollama_status(host: str, api_key: str, model: str) -> dict:
    return ollama_status(host, api_key, model)


def _context() -> dict:
    """Danışmana iletilecek pano bağlamı: kullanıcının en son incelediği ürün ve müşteri."""
    bp = get_engine()
    ctx = {"Referans tarih": "31.12.2020 (veri yalnızca 2020'yi kapsar)",
           "Önceki sayfa": st.session_state.get("last_page")}
    sc = st.session_state.get("selected_product")
    if sc is not None and sc in bp.portfolio.index:
        ctx["Seçili ürün"] = f"{bp.products.loc[sc, 'Description']} (StockCode {sc}, DNA: {bp.portfolio.loc[sc, 'DNA']})"
    cust = st.session_state.get("selected_customer_calls") or st.session_state.get("selected_customer")
    if cust:
        ctx["Seçili müşteri"] = cust
    if st.session_state.get("_calls_method"):
        ctx["Sıradaki 10 Arama puanlama yöntemi"] = st.session_state["_calls_method"]
    return ctx


def page_advisor(bp: BookPulse):
    md(ui.page_header("Yapay Zekâ Satış Danışmanı",
                      "Panodaki analizlere doğal dilde soru sorun. Danışman yalnızca BookPulse analiz fonksiyonlarını çağırır; "
                      "sayılar gerçek hesaplamalardan gelir. İncelediğiniz ürün ve müşteri otomatik olarak bağlama eklenir.",
                      eyebrow="Karar Destek"))
    md(ui.section("Dil modeli bağlantısı", "ücretsiz yerel Ollama veya kendi API anahtarınızla OpenAI / Claude / Gemini"))
    cfg = ollama_settings()
    status = _ollama_status(cfg["host"], cfg["api_key"], cfg["model"])
    with_keys = [p_ for p_ in PROVIDERS if p_ != "ollama" and provider_key(p_)]
    on_cloud = running_on_cloud()
    default_prov = ("ollama" if status["step"] == "ready" else with_keys[0] if with_keys
                    else "gemini" if on_cloud else "ollama")
    c1, c2, c3, c4 = st.columns([1.3, 1.6, 1.2, 0.9])
    prov_order = (["gemini", "openai", "claude", "ollama"] if on_cloud else list(PROVIDERS))
    prov_label = (lambda p_: "Ollama (yalnızca yerel kurulumda)" if p_ == "ollama" and on_cloud else PROVIDERS[p_]["label"])
    provider = keep_select(c1, "Sağlayıcı", prov_order, "ai_provider", default=default_prov,
                           format_func=prov_label, help=ui.GLOSSARY["dil modeli"])
    spec = PROVIDERS[provider]
    api_key, model, connected, host = "", spec["default_model"], False, cfg["host"]
    if provider == "ollama":
        host = keep_text(c2, "Ollama adresi", "ollama_host", cfg["host"])
        status = _ollama_status(host, cfg["api_key"], cfg["model"]) if host != cfg["host"] else status
        connected = status["step"] == "ready"
        if connected:
            model = keep_select(c3, "Model", status["models"], "ollama_model", default=status["model"])
        else:
            model = status["model"]
            c3.text_input("Model", value=model, disabled=True)
        api_key = cfg["api_key"]
    else:
        secret = provider_key(provider)
        sk = f"_key_{provider}"
        typed = c2.text_input("API anahtarı" + (" (Secrets'ta tanımlı)" if secret else ""), type="password",
                              value=st.session_state.get(sk, ""), key=f"w{sk}",
                              help=ui.GLOSSARY["api anahtarı"] + " Yalnızca bu oturumda kullanılır, kaydedilmez. " + spec["help"])
        st.session_state[sk] = typed
        api_key = typed.strip() or secret
        model = keep_text(c3, "Model", f"model_{provider}",
                          os.getenv(spec["model_env"], "").strip() or spec["default_model"])
        connected = bool(api_key)
    c4.write("")
    if connected and c4.button("Bağlantıyı test et"):
        try:
            st.success("Bağlantı çalışıyor: " + ping_provider(provider, model, api_key, host)[:60])
        except Exception as e:
            st.error("Bağlantı hatası: " + error_detail(e)[:400])

    ctx = _context()
    chips = " ".join(f'<span class="bp-chip">{k}: {v}</span>' for k, v in ctx.items()
                     if v and k in ("Seçili ürün", "Seçili müşteri"))
    md(f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">'
       f'<span class="bp-chip"><span class="dot" style="background:{P["green"] if connected else P["red"]}"></span>'
       f'{spec["label"] if connected else "Dil modeli bağlı değil"}'
       f'{" · " + model if connected else ""}</span>{chips}</div>')

    if not connected:
        if provider == "ollama" and on_cloud:
            md(ui.insight("Bu pano Streamlit Cloud sunucusunda çalışıyor; sunucu sizin bilgisayarınızdaki Ollama'ya "
                          "erişemez. Canlı panoda yukarıdan <b>Gemini, OpenAI veya Claude</b> seçip kendi API anahtarınızı "
                          "girin. Ollama'yı kullanmak için projeyi GitHub'dan indirip kendi bilgisayarınızda çalıştırın "
                          "(adımlar aşağıda).", tag="Bulut ortamı", risk=True))
            with st.expander("Yerel kurulum: Ollama ile ücretsiz kullanım"):
                st.markdown(setup_guide(status))
        elif provider == "ollama":
            md(ui.section("Ollama kurulumu", "veya yukarıdan kendi API anahtarınızla OpenAI / Claude / Gemini seçin"))
            st.markdown(setup_guide(status))
            if st.button("Tekrar kontrol et", type="primary"):
                _ollama_status.clear()
                rerun()
        else:
            md(ui.insight(f"{spec['label']} kullanmak için yukarıdaki alana API anahtarınızı girin ({spec['help']}). "
                          "Anahtar yalnızca bu oturumda kullanılır. Streamlit Cloud'da kalıcı kullanım için "
                          f"<code>{spec['key_env']}</code> değerini <i>App settings → Secrets</i> bölümüne ekleyin.",
                          tag="Anahtar gerekli"))
        st.divider()
        if not st.checkbox("Dil modeli olmadan dene: anahtar kelime modu (yalnızca tablo ve grafik)"):
            return
    elif provider != "ollama":
        md(ui.insight("Bulut modeli seçildiğinde sorunuz ve ilgili analiz sonuçları (müşteri kodları dahil) seçilen "
                      "sağlayıcıya gönderilir. Şirket verisi için paylaşım iznini kontrol edin.", tag="Veri notu"))
    advisor = SalesAdvisor(bp, provider=provider, model=model, api_key=api_key, host=host,
                           available=connected, context=ctx)
    _advisor_chat(advisor)


def _advisor_chat(advisor: SalesAdvisor):
    st.session_state.setdefault("chat", [])
    if not st.session_state["chat"]:
        md(ui.section("Örnek sorular", "tıklayarak deneyin"))
    cols = st.columns(4)
    clicked = None
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 4].button(ex, key=f"ex{i}"):
            clicked = ex
    if st.sidebar.button("Sohbeti temizle"):
        st.session_state["chat"] = []

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for call in msg.get("tools", []):
                _render_tool_result(*call)

    question = st.chat_input("Sorunuzu yazın…") or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat"][-8:]]
        with st.chat_message("assistant"):
            with st.spinner("Analiz ediliyor…"):
                try:
                    reply = advisor.ask(question, history=history)
                except Exception as e:
                    st.error(f"Beklenmeyen hata: {e}")
                    return
            st.markdown(reply.text)
            for call in reply.tool_calls:
                _render_tool_result(*call)
        st.session_state["chat"] += [{"role": "user", "content": question},
                                     {"role": "assistant", "content": reply.text, "tools": reply.tool_calls}]


# ===================================================================================
PAGE_FUNCS = {"Yönetici Paneli": page_executive, "Portföy Optimizasyonu": page_portfolio,
              "Sıradaki 10 Arama": page_calls, "Müşteri DNA": page_customers, "Sepet Analizi": page_basket,
              "Talep Tahmini": page_forecast, "Yapay Zekâ Satış Danışmanı": page_advisor}


def _scroll_top():
    try:
        import streamlit.components.v1 as components
        components.html("<script>const d=window.parent.document;"
                        "for(const sel of ['[data-testid=\"stMain\"]','section.main','[data-testid=\"stAppViewContainer\"]']){"
                        "const el=d.querySelector(sel); if(el){el.scrollTo(0,0);}} window.parent.scrollTo(0,0);</script>",
                        height=0)
    except Exception:
        pass


def main():
    page = sidebar()
    if not data_available():
        md(ui.page_header("BookPulse", "Veri bulunamadı."))
        st.error(f"`data/` klasöründe `{config.CLEAN_FILE.name}` veya `{config.RAW_XLSX.name}` bulunamadı.")
        return
    bp = get_engine()
    if st.session_state.get("_rendered_page") != page:
        st.session_state["_rendered_page"] = page
        _scroll_top()
    try:
        PAGE_FUNCS[page](bp)
    except Exception as e:
        if type(e).__name__ in ("RerunException", "StopException", "Rerun"):
            raise
        st.error("Bu sayfa yüklenirken bir hata oluştu. Lütfen sayfayı yenileyin veya başka bir sayfaya geçin.")
        with st.expander("Hata ayrıntısı"):
            st.exception(e)


main()
