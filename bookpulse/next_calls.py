"""Sıradaki 10 Arama v2 – Akıllı müşteri arama sistemi.

v1 (homework1.ipynb) yalnızca tek bir ürün için kategori miktarı / sipariş sayısı /
yakınlık ile puanlıyordu. v2:
  * Her Çok Satan – Düzenli Talep ve Çok Satan – Değişken Talep ürünü için ayrı liste üretir (batch).
  * Kategori payı, sipariş sıklığı, ürün çeşitliliği ve "benzer ürün" (ürünle birlikte
    alınan ürünler, item-based CF) sinyallerini ekler.
  * Uzun süredir alışveriş yapmayan müşterileri ayrıca işaretler.
  * Her öneriye okunabilir gerekçe ve aksiyon önerisi ekler.
  * Ters yön: bir müşteriye hangi ürünlerin önerileceği (benzer müşteriler, user-based CF).
  * backtest(): Oca–Eyl verisiyle kurulur, Eki–Ara'da gerçekleşen alımlarla ölçülür.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from . import config
from .data import product_lookup

FEATURE_LABELS = {
    "category_qty": "ilgili kategoride yüksek alım hacmi",
    "category_share": "alımlarının önemli kısmı bu kategoride",
    "frequency": "bu kategoride sık sipariş veriyor",
    "diversity": "kategoride çok sayıda farklı ürün alıyor",
    "similar_items": "bu ürünle birlikte alınan ürünleri satın almış",
    "recency": "yakın zamanda alışveriş yapmış",
    "category_orders": "bu kategoride çok sipariş vermiş",
    "category_recency": "bu kategoriden yakın zamanda almış",
}

V1_WEIGHTS = {"category_qty": 0.50, "category_orders": 0.30, "category_recency": 0.20}


class NextCallsEngine:
    def __init__(self, df: pd.DataFrame, portfolio: pd.DataFrame | None = None,
                 weights: dict | None = None, ref_date=None):
        self.df = df
        self.portfolio = portfolio
        self.weights = weights or config.NEXT_CALLS_WEIGHTS
        self.ref_date = pd.Timestamp(ref_date) if ref_date is not None else df["InvoiceDate"].max()
        self.products = product_lookup(df)
        self._prepare()

    # ------------------------------------------------------------------
    def _prepare(self):
        df = self.df
        self.cust_ids = np.sort(df["CustomerID"].unique())
        self.prod_ids = np.sort(df["StockCode"].unique())
        self.c_idx = pd.Series(np.arange(len(self.cust_ids)), index=self.cust_ids)
        self.p_idx = pd.Series(np.arange(len(self.prod_ids)), index=self.prod_ids)

        pairs = df[["CustomerID", "StockCode"]].drop_duplicates()
        rows = self.c_idx[pairs["CustomerID"]].to_numpy()
        cols = self.p_idx[pairs["StockCode"]].to_numpy()
        self.B = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.cust_ids), len(self.prod_ids)))
        self.BT = self.B.T.tocsr()
        self.prod_buyers = np.asarray(self.B.sum(axis=0)).ravel()

        self.cat_stats = df.groupby(["Category", "CustomerID"]).agg(
            cat_qty=("Quantity", "sum"), cat_orders=("InvoiceID", "nunique"),
            cat_products=("StockCode", "nunique"), cat_last=("InvoiceDate", "max"),
        )
        self.cust_stats = df.groupby("CustomerID").agg(
            total_qty=("Quantity", "sum"), last_purchase=("InvoiceDate", "max"),
            first_purchase=("InvoiceDate", "min"),
        )
        first = df.drop_duplicates("CustomerID").set_index("CustomerID")
        self.cust_stats["City"] = first["City"]
        self.cust_stats["CustomerType"] = first["CustomerType"]
        months = ((self.ref_date - self.cust_stats["first_purchase"]).dt.days / 30.4).clip(lower=1)
        self.cust_stats["active_months"] = months
        self.cust_stats["days_since"] = (self.ref_date - self.cust_stats["last_purchase"]).dt.days

    # ------------------------------------------------------------------
    def similar_products(self, stock_code, k: int = 30, min_co: int = 3) -> pd.Series:
        """Aynı müşterilerce alınan ürünler (cosine benzerliği)."""
        j = self.p_idx[stock_code]
        co = np.asarray((self.BT @ self.BT[j].T).todense()).ravel()
        co[j] = 0
        co[co < min_co] = 0
        sim = co / np.sqrt(self.prod_buyers * max(self.prod_buyers[j], 1) + 1e-9)
        top = np.argsort(-sim)[:k]
        top = top[sim[top] > 0]
        return pd.Series(sim[top], index=self.prod_ids[top], name="similarity")

    def _features(self, stock_code) -> pd.DataFrame:
        category = self.products.loc[stock_code, "Category"]
        buyers = set(self.cust_ids[self.BT[self.p_idx[stock_code]].indices])

        sims = self.similar_products(stock_code)
        sim_score = np.zeros(len(self.cust_ids))
        if len(sims):
            cols = self.p_idx[sims.index].to_numpy()
            sim_score = self.B[:, cols] @ sims.to_numpy()
        sim_s = pd.Series(sim_score, index=self.cust_ids)

        cat = (self.cat_stats.loc[category] if category in self.cat_stats.index.get_level_values(0)
               else pd.DataFrame(columns=self.cat_stats.columns))
        cand_ids = set(cat.index) | set(sim_s[sim_s > 0].index)
        cand_ids -= buyers
        if not cand_ids:
            return pd.DataFrame()

        f = pd.DataFrame(index=sorted(cand_ids))
        f.index.name = "CustomerID"
        f = f.join(cat).join(self.cust_stats)
        f[["cat_qty", "cat_orders", "cat_products"]] = f[["cat_qty", "cat_orders", "cat_products"]].fillna(0)
        f["category_qty"] = f["cat_qty"]
        f["category_share"] = f["cat_qty"] / f["total_qty"].clip(lower=1)
        f["frequency"] = f["cat_orders"] / f["active_months"]
        f["diversity"] = f["cat_products"]
        f["similar_items"] = sim_s.reindex(f.index).fillna(0)
        f["recency"] = -f["days_since"]
        f["category_orders"] = f["cat_orders"]
        f["category_recency"] = -(self.ref_date - f["cat_last"]).dt.days.fillna(9999)
        f.attrs.update(category=category, n_buyers=len(buyers), n_similar=len(sims))
        return f

    def _score(self, f: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
        w = {k: v for k, v in (weights or self.weights).items() if v > 0}
        total = sum(w.values()) or 1
        w = {k: v / total for k, v in w.items()}            
        ranks = pd.DataFrame({k: f[k].rank(pct=True) for k in w})
        for k in ["category_qty", "category_share", "frequency", "diversity", "similar_items", "category_orders"]:
            if k in ranks:
                ranks.loc[f[k] <= 0, k] = 0
        contrib = ranks.mul(pd.Series(w))
        f = f.copy()
        f["Score"] = (100 * contrib.sum(axis=1)).round(1)
        f["_top"] = contrib.apply(lambda r: list(r.sort_values(ascending=False).index[:3]), axis=1)
        return f

    # ------------------------------------------------------------------
    def recommend(self, stock_code, n: int = 10, explain: bool = True, weights: dict | None = None) -> pd.DataFrame:
        """Bir ürün için en uygun n müşteriyi döndürür. weights verilirse varsayılan ağırlıkların yerine kullanılır
        (ör. V1_WEIGHTS veya kullanıcının kenar çubuğunda belirlediği ağırlıklar)."""
        f = self._features(stock_code)
        if f.empty:
            return pd.DataFrame()
        f = self._score(f, weights).sort_values("Score", ascending=False).head(n)
        dna = None
        if self.portfolio is not None and stock_code in self.portfolio.index:
            dna = self.portfolio.loc[stock_code, "DNA"]

        out = pd.DataFrame({
            "CustomerID": f.index,
            "StockCode": stock_code,
            "Product": self.products.loc[stock_code, "Description"],
            "DNA": dna,
            "Score": f["Score"].to_numpy(),
            "Category_Qty": f["cat_qty"].to_numpy(),
            "Category_Orders": f["cat_orders"].astype(int).to_numpy(),
            "Distinct_Products": f["cat_products"].astype(int).to_numpy(),
            "Similar_Score": f["similar_items"].round(2).to_numpy(),
            "Days_Since_Last": f["days_since"].astype(int).to_numpy(),
            "City": f["City"].to_numpy(),
        })
        out["Dormant"] = out["Days_Since_Last"] > config.DORMANT_DAYS
        lo, hi = self.evidence_thresholds
        sim = f["similar_items"].to_numpy()
        out["Priority"] = np.select(
            [out["Dormant"], sim >= hi, sim >= lo],
            ["Geri Kazanım Araması", "Hemen Ara", "Teklif Sun"], default="Takip Et")
        if explain:
            out["Reason"] = [self._reason(t, d, dna) for t, d in zip(f["_top"], out["Dormant"])]
        return out.reset_index(drop=True)

    @property
    def evidence_thresholds(self) -> tuple[float, float]:
        """(Teklif Sun, Hemen Ara) eşikleri: en çok müşteriye ulaşan 300 üründen 30'unun ilk-10 listelerinde
        'benzer ürün' sinyalinin %40 ve %75 yüzdelikleri. Ürüne göre değişmez, veri dönemine göre ölçeklenir."""
        if getattr(self, "_ev_thr", None) is None:
            top = self.prod_ids[np.argsort(-self.prod_buyers)[:300:10]]
            vals = []
            for sc in top:
                f = self._features(sc)
                if not f.empty:
                    vals.extend(self._score(f, None).sort_values("Score", ascending=False).head(10)["similar_items"])
            v = np.asarray(vals, float) if vals else np.zeros(1)
            self._ev_thr = (float(np.quantile(v, 0.40)), float(np.quantile(v, 0.75)))
        return self._ev_thr

    def _reason(self, top, dormant, dna) -> str:
        parts = [FEATURE_LABELS[t] for t in top]
        txt = "Bu müşteri " + ", ".join(parts) + "; önerilen ürünü henüz satın almamış."
        if dormant:
            txt += f" {config.DORMANT_DAYS}+ gündür alışveriş yapmıyor – geri kazanım fırsatı."
        if dna == "Çok Satan – Değişken Talep":
            txt += " Talep dalgalı: sezon/kampanya teklifiyle ara."
        elif dna == "Çok Satan – Düzenli Talep":
            txt += " Sürekli satan ürün: standart teklif sun."
        return txt

    def batch(self, stock_codes, n: int = 10, progress=None) -> pd.DataFrame:
        """Birden çok ürün için öneri tablosu (uzun format)."""
        frames = []
        for i, sc in enumerate(stock_codes):
            r = self.recommend(sc, n=n, explain=True)
            if not r.empty:
                frames.append(r)
            if progress:
                progress((i + 1) / len(stock_codes))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def customer_opportunities(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Toplu listeyi müşteri bazında birleştirir: her müşteri için tek kart (birden çok ürün fırsatı).
        Not: puanlar ürün içinde göreli sıralamadır; farklı ürünlerin puanları doğrudan kıyaslanmamalı.
        Bu yüzden müşteriler önce fırsat sayısına, sonra müşterinin yıllık cirosuna göre sıralanır."""
        if batch.empty:
            return batch
        rev = self.df.groupby("CustomerID")["Revenue"].sum()
        prio_rank = {"Hemen Ara": 0, "Geri Kazanım Araması": 1, "Teklif Sun": 2, "Takip Et": 3}
        g = batch.assign(_p=batch["Priority"].map(prio_rank)).sort_values("Score", ascending=False).groupby("CustomerID")
        out = pd.DataFrame({
            "Opportunities": g.size(),
            "Top_Products": g["Product"].apply(lambda s_: " · ".join(s_.head(3))),
            "Median_Score": g["Score"].median().round(1),
            "Priority": g["_p"].min().map({v: k for k, v in prio_rank.items()}),
            "Days_Since_Last": g["Days_Since_Last"].first(),
            "City": g["City"].first(),
        })
        out["Customer_Revenue"] = rev.reindex(out.index).round(0)
        return out.sort_values(["Opportunities", "Customer_Revenue"], ascending=False)

    # ------------------------------------------------------------------
    def products_for_customer(self, customer_id, n: int = 10, dna_filter=("Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep"),
                              k_neighbors: int = 50) -> pd.DataFrame:
        """Ters yön: bir müşteriye önerilecek ürünler (benzer müşterilerin aldıkları)."""
        if customer_id not in self.c_idx.index:
            return pd.DataFrame()
        i = self.c_idx[customer_id]
        row = self.B[i]
        overlap = np.asarray((self.B @ row.T).todense()).ravel()
        sizes = np.asarray(self.B.sum(axis=1)).ravel()
        sim = overlap / np.sqrt(sizes * max(sizes[i], 1) + 1e-9)
        sim[i] = 0
        nb = np.argsort(-sim)[:k_neighbors]
        nb = nb[sim[nb] > 0]
        if len(nb) == 0:
            return pd.DataFrame()
        score = (sparse.csr_matrix(sim[nb]) @ self.B[nb]).toarray().ravel()
        count = np.asarray(self.B[nb].sum(axis=0)).ravel()
        score[row.indices] = 0                              
        cand = pd.DataFrame({"StockCode": self.prod_ids, "CF_Score": score, "Neighbors_Bought": count})
        cand = cand[cand["CF_Score"] > 0].set_index("StockCode").join(self.products)
        if self.portfolio is not None:
            cand = cand.join(self.portfolio[["DNA", "Performance_Score"]])
            if dna_filter:
                cand = cand[cand["DNA"].isin(dna_filter)]
        cat_aff = self.df[self.df["CustomerID"] == customer_id].groupby("Category")["Quantity"].sum()
        cat_aff = cat_aff / cat_aff.sum()
        cand["Category_Affinity"] = cand["Category"].map(cat_aff).fillna(0)
        cand["Score"] = 100 * (0.7 * cand["CF_Score"].rank(pct=True) + 0.3 * cand["Category_Affinity"].rank(pct=True))
        cand = cand.sort_values("Score", ascending=False).head(n).reset_index()
        cand["Reason"] = [
            f"Benzer {int(c)} müşteri bu ürünü almış; müşterinin '{cat}' kategorisindeki payı %{100*a:.0f}."
            for c, cat, a in zip(cand["Neighbors_Bought"], cand["Category"], cand["Category_Affinity"])]
        cols = ["StockCode", "Description", "Category", "DNA", "Score", "Neighbors_Bought", "Category_Affinity", "Reason"]
        return cand[[c for c in cols if c in cand.columns]].round({"Score": 1, "Category_Affinity": 3})

    def customer_portfolio(self, customer_id) -> pd.DataFrame:
        """Müşteri Portföy DNA'sı: müşterinin alımlarının DNA gruplarına dağılımı, şirket geneliyle karşılaştırmalı."""
        if self.portfolio is None:
            return pd.DataFrame()
        from .portfolio import DNA_ORDER
        d = self.df[self.df["CustomerID"] == customer_id].merge(
            self.portfolio[["DNA"]], left_on="StockCode", right_index=True)
        g = d.groupby("DNA").agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"),
                                 Products=("StockCode", "nunique"), Orders=("InvoiceID", "nunique"))
        g["Customer_Share_%"] = 100 * g["Quantity"] / g["Quantity"].sum()
        company = self.portfolio.groupby("DNA")["Total_Qty"].sum()
        g["Company_Share_%"] = 100 * company / company.sum()
        g["Difference_pp"] = g["Customer_Share_%"] - g["Company_Share_%"]
        return g.reindex([x for x in DNA_ORDER if x in g.index]).round(2)

    def customer_products(self, customer_id, dna=None, n: int = 20) -> pd.DataFrame:
        """Müşterinin en çok aldığı ürünler (isteğe bağlı DNA grubuyla filtreli)."""
        d = self.df[self.df["CustomerID"] == customer_id]
        t = d.groupby("StockCode").agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"),
                                       Orders=("InvoiceID", "nunique"), Last_Sale=("InvoiceDate", "max"))
        t = t.join(self.products)
        if self.portfolio is not None:
            t = t.join(self.portfolio[["DNA"]])
            if dna:
                t = t[t["DNA"] == dna]
        return t.sort_values("Quantity", ascending=False).head(n)

    def why_customer(self, customer_id, stock_code) -> dict:
        """'Neden bu müşteri?' – öneriyi destekleyen kanıtlar."""
        category = self.products.loc[stock_code, "Category"]
        d = self.df[self.df["CustomerID"] == customer_id]
        top_cats = d.groupby("Category")["Quantity"].sum().nlargest(5)
        sims = self.similar_products(stock_code, k=30)
        bought_similar = sorted(set(d["StockCode"]) & set(sims.index))
        buyers = self.prod_buyers[self.p_idx[stock_code]]
        cat_qty = float(d.loc[d["Category"] == category, "Quantity"].sum())
        return {
            "category": category,
            "category_share_%": round(float(100 * cat_qty / max(d["Quantity"].sum(), 1)), 1),
            "top_categories": {k: float(v) for k, v in top_cats.items()},
            "similar_products_bought": [self.products.loc[s, "Description"] for s in bought_similar[:8]],
            "n_similar_products_bought": len(bought_similar),
            "product_buyers": int(buyers),
            "already_bought": customer_id in set(self.cust_ids[self.BT[self.p_idx[stock_code]].indices]),
            "days_since_last": int(self.cust_stats.loc[customer_id, "days_since"]),
        }

    def customer_flags(self) -> pd.DataFrame:
        """Uyuyan müşteriler listesi."""
        s = self.cust_stats.copy()
        s["Dormant"] = s["days_since"] > config.DORMANT_DAYS
        return s.sort_values("total_qty", ascending=False)


# ---------------------------------------------------------------------------
def v1_score(df: pd.DataFrame, stock_code, n=10) -> list:
    """Ekibin ilk versiyonu (karşılaştırma için): kategori miktarı %50, sipariş %30, yakınlık %20."""
    cat = df.loc[df["StockCode"] == stock_code, "Category"].mode().iloc[0]
    cand = df[df["Category"] == cat]
    buyers = df.loc[df["StockCode"] == stock_code, "CustomerID"].unique()
    cand = cand[~cand["CustomerID"].isin(buyers)]
    if cand.empty:
        return []
    s = cand.groupby("CustomerID").agg(q=("Quantity", "sum"), o=("InvoiceID", "nunique"), last=("InvoiceDate", "max"))
    rec = (df["InvoiceDate"].max() - s["last"]).dt.days
    s["score"] = 0.5 * s["q"].rank(pct=True) + 0.3 * s["o"].rank(pct=True) + 0.2 * rec.rank(ascending=False, pct=True)
    return list(s.sort_values("score", ascending=False).head(n).index)


def backtest(df: pd.DataFrame, cutoff="2020-09-30", n_products: int = 300, n: int = 10,
             min_buyers: int = 10, seed: int = 42, portfolio_builder=None,
             weights: dict | None = None, horizon_end=None) -> dict:
    """Geriye dönük test.

    Model `cutoff` tarihine kadar olan veriyle kurulur. Train döneminde Çok Satan – Düzenli Talep /
    Çok Satan – Değişken Talep olan örnek ürünler için top-n müşteri önerilir; öneri, müşteri o ürünü
    test döneminde (cutoff sonrası) gerçekten aldıysa "isabet" sayılır.
    Karşılaştırma: v1 puanlaması ve rastgele kategori müşterisi.
    """
    from .portfolio import build_portfolio
    cutoff = pd.Timestamp(cutoff)
    end = pd.Timestamp(horizon_end) if horizon_end is not None else df["InvoiceDate"].max()
    train = df[df["InvoiceDate"] <= cutoff]
    test = df[(df["InvoiceDate"] > cutoff) & (df["InvoiceDate"] <= end)]
    p, _ = (portfolio_builder or build_portfolio)(train)
    eng = NextCallsEngine(train, p, weights=weights)

    pool = p[p["DNA"].isin(["Çok Satan – Düzenli Talep", "Çok Satan – Değişken Talep"]) & (p["Customers"] >= min_buyers)]
    rng = np.random.default_rng(seed)
    sample = rng.choice(pool.index.to_numpy(), size=min(n_products, len(pool)), replace=False)

    test_pairs = set(zip(test["CustomerID"], test["StockCode"]))
    cat_customers = train.groupby("Category")["CustomerID"].unique()

    hits = {"v2": 0, "v1": 0, "random": 0}
    any_hit = {"v2": 0, "v1": 0, "random": 0}
    total = {"v2": 0, "v1": 0, "random": 0}
    for sc in sample:
        buyers = set(train.loc[train["StockCode"] == sc, "CustomerID"])
        lists = {}
        r = eng.recommend(sc, n=n, explain=False)
        lists["v2"] = list(r["CustomerID"]) if not r.empty else []
        lists["v1"] = v1_score(train, sc, n)
        cat = eng.products.loc[sc, "Category"]
        pool_c = [c for c in cat_customers.get(cat, []) if c not in buyers]
        lists["random"] = list(rng.choice(pool_c, size=min(n, len(pool_c)), replace=False)) if pool_c else []
        for k, lst in lists.items():
            h = sum((c, sc) in test_pairs for c in lst)
            hits[k] += h
            any_hit[k] += int(h > 0)
            total[k] += len(lst)

    res = pd.DataFrame({
        "Precision@10_%": {k: 100 * hits[k] / max(total[k], 1) for k in hits},
        "Hit_Rate_%": {k: 100 * any_hit[k] / len(sample) for k in hits},
        "Hits": hits,
    }).round(2)
    res.index = res.index.map({"v2": "Sıradaki 10 Arama v2", "v1": "v1 (ekibin ilk modeli)", "random": "Rastgele kategori müşterisi"})
    return {"table": res, "n_products": len(sample), "cutoff": str(cutoff.date())}
