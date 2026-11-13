"""Gerçek API anahtarlarıyla dil modeli sağlayıcılarını uçtan uca test eder.

Anahtarları ortam değişkeni olarak verin (yalnızca bu komut için, dosyaya yazılmaz):
    GEMINI_API_KEY=... python tools/check_providers.py
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python tools/check_providers.py

Her sağlayıcı için bir basit ve bir çok araçlı soru sorulur; çağrılan analiz araçları ve cevabın ilk satırı yazdırılır.
Ollama, bilgisayarda çalışıyorsa otomatik test edilir.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpulse import advisor  # noqa: E402
from bookpulse.engine import BookPulse  # noqa: E402

QUESTIONS = [
    "Toplam ciro ve müşteri sayısı nedir?",
    "Son üç ayda düşen kategoriler hangileri ve Şeker Portakalı için hangi müşterileri aramalıyım?",
]

bp = BookPulse()
tested = 0
for prov, spec in advisor.PROVIDERS.items():
    if prov == "ollama":
        st_ = advisor.ollama_status()
        if st_["step"] != "ready":
            print(f"[{prov}] atlandı: {st_['step']}")
            continue
        adv = advisor.SalesAdvisor(bp, provider=prov, model=st_["model"], available=True)
    else:
        key = advisor.provider_key(prov)
        if not key:
            print(f"[{prov}] atlandı: {spec['key_env']} tanımlı değil")
            continue
        adv = advisor.SalesAdvisor(bp, provider=prov, api_key=key, available=True,
                                   context={"Seçili müşteri": "A01913"})
    tested += 1
    for q in QUESTIONS:
        t = time.time()
        rep = adv.ask(q)
        tools = [c[0] for c in rep.tool_calls]
        status = "HATA" if rep.error or rep.mode == "fallback" else "OK"
        print(f"[{prov}] {status} ({time.time() - t:.1f} sn) araçlar={tools}")
        print("   ", (rep.error or rep.text).strip().splitlines()[0][:160])
print(f"\n{tested} sağlayıcı test edildi.")
