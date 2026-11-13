import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpulse import advisor
from bookpulse.engine import BookPulse

ANSWER = "Sözlük kategorisi en çok düşen kategori."
calls = {"n": 0}


def fake_request(host, path, api_key="", payload=None, timeout=180):
    assert path == "/api/chat" and payload["stream"] is False and payload["tools"] == advisor.OLLAMA_TOOLS
    calls["n"] += 1
    if calls["n"] == 1:
        return {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "category_trend", "arguments": {"months": 3, "n": 3}}},
            {"function": {"name": "next_calls_for_product", "arguments": '{"product": "Şeker Portakalı", "n": 3}'}}]}}
    tool_msgs = [m for m in payload["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and {m["tool_name"] for m in tool_msgs} == {"category_trend", "next_calls_for_product"}
    for m in tool_msgs:
        assert "error" not in json.loads(m["content"]), m["content"][:200]
    return {"message": {"role": "assistant", "content": ANSWER}}


def failing_request(*a, **k):
    raise advisor.OllamaError("HTTP 404: model 'yok:1b' not found", 404)


bp = BookPulse()
advisor._request = fake_request
adv = advisor.SalesAdvisor(bp, host="http://x", api_key="", model="qwen2.5:7b", available=True)
rep = adv.ask("Son 3 ayda düşen kategoriler ve Şeker Portakalı için kimi aramalıyım?")
assert rep.text == ANSWER and [c[0] for c in rep.tool_calls] == ["category_trend", "next_calls_for_product"]
print("Ollama araç döngüsü OK:", rep.text)

advisor._request = failing_request
rep = advisor.SalesAdvisor(bp, host="http://x", model="yok:1b", available=True).ask("Son üç ayda düşen kategoriler")
assert "bulunamadı" in rep.text and rep.tool_calls and rep.error
print("Hata durumunda yedek mod OK")

rep = advisor.SalesAdvisor(bp, host="http://127.0.0.1:9", available=None).ask("En iyi 5 şehir")
assert rep.mode == "fallback" and rep.tool_calls[0][0] == "top_entities"
print("Ollama yokken yedek mod OK")

st_ = advisor.ollama_status(host="http://127.0.0.1:9")
assert st_["step"] in ("not_installed", "not_running") and "ollama.com/download" in advisor.setup_guide(st_)
print("Kurulum rehberi OK:", st_["step"])


from types import SimpleNamespace as NS 


class FakeOpenAI:
    def __init__(self):
        self.n = 0

    def create(self, model, messages, tools):
        self.n += 1
        assert messages[0]["role"] == "system" and "A01913" in messages[0]["content"]    
        if self.n == 1:
            tc = NS(id="c1", type="function", function=NS(name="customer_portfolio_dna", arguments='{"customer_id": "A01913"}'),
                    model_extra={"extra_content": {"google": {"thought_signature": "sig"}}})
            return NS(choices=[NS(message=NS(content=None, tool_calls=[tc], model_extra={}))])
        assert messages[-2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig"
        return NS(choices=[NS(message=NS(content=ANSWER, tool_calls=None, model_extra={}))])


class FakeClaude:
    def __init__(self):
        self.n = 0

    def create(self, model, max_tokens, system, tools, messages):
        self.n += 1
        if self.n == 1:
            return NS(stop_reason="tool_use", content=[NS(type="tool_use", id="t1", name="get_kpis", input={})])
        assert messages[-1]["content"][0]["type"] == "tool_result"
        return NS(stop_reason="end_turn", content=[NS(type="text", text=ANSWER)])


for prov, client in [("gemini", NS(chat=NS(completions=FakeOpenAI()))), ("openai", NS(chat=NS(completions=FakeOpenAI()))),
                     ("claude", NS(messages=FakeClaude()))]:
    a = advisor.SalesAdvisor(bp, provider=prov, api_key="x", available=True, context={"Seçili müşteri": "A01913"})
    a.client, a.available = client, True
    rep = a.ask("Bu müşteriyi neden aramalıyım?")
    assert rep.text == ANSWER and rep.tool_calls, (prov, rep.text)
    print(f"{prov} sağlayıcı döngüsü OK")
assert all("properties" in t["function"].get("parameters", {"properties": 1}) for t in advisor.OPENAI_TOOLS)
