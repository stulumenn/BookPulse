"""BookPulse analiz motoru: tüm modüllerin sonuçlarını tembel (lazy) hesaplayıp tutar.
Streamlit uygulaması ve Yapay Zekâ Satış Danışmanı aynı nesneyi kullanır."""
from __future__ import annotations

from functools import cached_property

import pandas as pd

from . import basket, customer_dna, data, forecast, next_calls, portfolio


class BookPulse:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else data.load_clean()

    # --- temel ---------------------------------------------------------
    @cached_property
    def kpis(self) -> dict:
        return data.kpis(self.df)

    @cached_property
    def products(self) -> pd.DataFrame:
        return data.product_lookup(self.df)

    # --- Portfolio ------------------------------------------------------
    @cached_property
    def _portfolio(self):
        return portfolio.build_portfolio(self.df)

    @property
    def portfolio(self) -> pd.DataFrame:
        return self._portfolio[0]

    @property
    def thresholds(self) -> dict:
        return self._portfolio[1]

    # --- Sıradaki 10 Arama --------------------------------------------------
    @cached_property
    def calls(self) -> next_calls.NextCallsEngine:
        return next_calls.NextCallsEngine(self.df, self.portfolio)

    # --- Müşteri DNA ---------------------------------------------------
    @cached_property
    def _customers(self):
        rfm = customer_dna.rfm_table(self.df)
        return customer_dna.kmeans_segments(rfm)

    @property
    def customers(self) -> pd.DataFrame:
        return self._customers[0]

    @property
    def cluster_info(self) -> dict:
        return self._customers[1]

    # --- Sepet Analizi --------------------------------------------------
    @cached_property
    def product_rules(self) -> pd.DataFrame:
        return basket.association_rules(self.df, level="product")

    @cached_property
    def category_rules(self) -> pd.DataFrame:
        return basket.association_rules(self.df, level="category", min_support_count=50, min_lift=1.5)

    # --- Forecast -------------------------------------------------------
    @cached_property
    def weekly(self) -> pd.DataFrame:
        return forecast.weekly_series(self.df, level="Category", top_n=20)

    @cached_property
    def forecast_eval(self) -> dict:
        return forecast.evaluate(self.weekly)

    @cached_property
    def forecast_table(self) -> pd.DataFrame:
        return forecast.forecast(self.weekly, horizon=8, evaluation=self.forecast_eval)

    # --- yardımcılar ----------------------------------------------------
    def find_products(self, query: str, n: int = 10) -> pd.DataFrame:
        """Ürün adı / barkod ile arama."""
        q = str(query).strip()
        if q.isdigit() and int(q) in self.portfolio.index:
            return self.portfolio.loc[[int(q)]]
        mask = self.portfolio["Description"].str.contains(q, case=False, na=False, regex=False)
        return self.portfolio[mask].sort_values("Revenue", ascending=False).head(n)

    def resolve_product(self, query) -> int | None:
        hits = self.find_products(str(query), n=1)
        return int(hits.index[0]) if len(hits) else None
