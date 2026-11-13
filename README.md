# BookPulse – B2B Satış Analiz Platformu

BookPulse, satış verilerini inceleyerek ürün ve müşteri performansını daha anlaşılır hâle getiren bir analiz panelidir. Yöneticilerin satışları takip etmesine, öne çıkan eğilimleri görmesine ve daha bilinçli kararlar almasına yardımcı olur.

## Neler Sunar?

- **Genel Bakış:** Ciro, müşteri, kategori ve şehir bazında satış özeti. Tarih, şehir ve kategori filtreleri.
- **Ürün Analizi:** Ürünleri satış hacmi ve talep düzenine göre gruplandırır. Her grup için performans ve aksiyon önerileri sunar.
- **Müşteri Analizi:** Müşterileri alışveriş sıklığı, güncellik ve harcama değerlerine göre segmentlere ayırır. Takip edilmesi gereken müşterileri öne çıkarır.
- **Satış Önerileri:** Ürünlerle birlikte satın alınan ürünleri ve ilgili müşterileri analiz ederek satış ekibine aranabilecek müşteri listeleri sunar.
- **Sepet Analizi:** Birlikte satın alınan ürünleri belirler; çapraz satış ve paket önerileri oluşturur.
- **Talep Tahmini:** Geçmiş satışlara dayanarak kategori bazında önümüzdeki sekiz haftaya ilişkin tahminler üretir.

## Kullanılan Yöntemler

Analizlerde tanımlayıcı istatistikler, Pareto analizi, RFM müşteri segmentasyonu, K-Means kümeleme, birliktelik kuralları ve zaman serisi tahmin yöntemlerinden yararlanılır. Sonuçlar tablo ve grafiklerle sunulur.

## Proje Bağlantıları

- **Canlı Uygulama:** https://[bookpulse-4rkzhnlm8tjhaua7das4en.streamlit.app](https://bookpulse-orgsvjvdfj8lzmttrpfmyn.streamlit.app)
- **GitHub:** https://github.com/stulumenn/BookPulse

## Yerelde Çalıştırma

Python 3.11 veya 3.12 gereklidir.

```bash
git clone https://github.com/stulumenn/BookPulse.git
cd BookPulse
pip install -r requirements.txt
streamlit run streamlit_app.py
