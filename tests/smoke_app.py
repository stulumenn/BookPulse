import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CALLS = []


class Anything:
    def __init__(self, name="obj"):
        self._n = name

    def __getattr__(self, k):
        return Anything(f"{self._n}.{k}")

    def __call__(self, *a, **k):
        CALLS.append(self._n)
        return Anything(self._n + "()")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])

    def __setitem__(self, k, v):
        pass

    def __getitem__(self, k):
        return Anything(f"{self._n}[{k}]")


class SessionState(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Rerun(Exception):
    pass


def make_streamlit(page, buttons=(), radio_pick=None):
    st = types.ModuleType("streamlit")
    st.session_state = SessionState(page=page)
    st.secrets = {}

    def passthrough_decorator(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
    st.cache_resource = passthrough_decorator
    st.cache_data = passthrough_decorator

    class Container:
        def __getattr__(self, k):
            return getattr(st, k)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _cols(spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [Container() for _ in range(n)]

    def selectbox(label, options, index=0, format_func=str, **k):
        options = list(options)
        if not options:
            return None
        v = options[index or 0]
        format_func(v)
        return v

    def radio(label, options, index=0, format_func=str, **k):
        options = list(options)
        [format_func(o) for o in options]
        if radio_pick:
            for o in options:
                if radio_pick in str(o):
                    return o
        return options[index]

    def text_input(label, value="", **k):
        return value

    def button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary",
               disabled=False, use_container_width=False):
        CALLS.append(f"button:{label}")
        return any(b in label for b in buttons)

    def download_button(label, data, file_name=None, mime=None, key=None, help=None, on_click=None, args=None,
                        kwargs=None, *, type="secondary", disabled=False, use_container_width=False):
        return False

    def rerun():
        raise Rerun()

    def _noop(*a, **k):
        CALLS.append("st")
        return None

    for name in ["set_page_config", "title", "caption", "markdown", "write", "metric", "plotly_chart", "dataframe",
                 "info", "warning", "error", "subheader", "divider", "json", "spinner_", "success"]:
        setattr(st, name, _noop)
    def exception(e):
        raise e
    st.exception = exception
    st.columns = _cols
    st.empty = lambda: Container()
    st.container = lambda *a, **k: Container()
    st.tabs = lambda labels: [Container() for _ in labels]
    st.expander = lambda label, expanded=False: Container()
    st.chat_message = lambda name, avatar=None: Container()
    st.checkbox = lambda label, value=False, **k: value or any(b in label for b in buttons)
    st.spinner = lambda *a, **k: Container()
    st.selectbox = selectbox
    st.radio = radio
    st.text_input = text_input
    st.button = button
    st.download_button = download_button
    st.slider = lambda label, min_value=None, max_value=None, value=None, *a, **k: value if value is not None else min_value
    st.select_slider = lambda label, options, value=None, **k: value
    st.number_input = lambda label, mn=None, mx=None, value=None, *a, **k: value
    st.multiselect = lambda label, options, default=None, **k: list(default or [])
    st.toggle = lambda label, value=False, **k: value
    st.chat_input = lambda *a, **k: st.session_state.get("_chat_q")
    st.rerun = rerun
    st.column_config = Anything("column_config")
    st.sidebar = Container()
    return st


def run_page(page, buttons=(), chat_q=None, radio_pick=None, state=None):
    st = make_streamlit(page, buttons, radio_pick)
    st.session_state.update(state or {})
    if chat_q:
        st.session_state["_chat_q"] = chat_q
    sys.modules["streamlit"] = st
    for mod in ["plotly", "plotly.express", "plotly.graph_objects", "plotly.io"]:
        sys.modules[mod] = Anything(mod)
    sys.modules["plotly"].express = sys.modules["plotly.express"]
    try:
        runpy.run_path(str(ROOT / "streamlit_app.py"), run_name="__main__")
    except Rerun:
        return "rerun"
    return "ok"


if __name__ == "__main__":
    import time
    pages = ["Yönetici Paneli", "Portföy Optimizasyonu", "Sıradaki 10 Arama", "Müşteri DNA",
             "Sepet Analizi", "Talep Tahmini", "Yapay Zekâ Satış Danışmanı"]
    for pg in pages:
        t = time.time()
        print(f"{pg:25s}", run_page(pg), f"{time.time() - t:.1f}s")
    print("Danışman (Ollama yok, rehber)", run_page("Yapay Zekâ Satış Danışmanı"))
    print("Danışman (anahtar kelime modu)", run_page("Yapay Zekâ Satış Danışmanı", buttons=("Dil modeli olmadan dene",),
                                                    chat_q="Son üç ayda düşen kategoriler"))
    print("Portföy -> Arama geçişi   ", run_page("Portföy Optimizasyonu", buttons=("Sıradaki 10 Arama",)))
    print("Arama: ekibin ilk modeli  ", run_page("Sıradaki 10 Arama", radio_pick="Ekibin"))
    print("Arama: özel ağırlıklar    ", run_page("Sıradaki 10 Arama", radio_pick="Özel"))
    print("Arama -> yeni ürün döngüsü", run_page("Sıradaki 10 Arama", buttons=("Bu ürün için yeni liste",)))
    print("Panel -> Kaçan sipariş    ", run_page("Yönetici Paneli", buttons=("Listeyi aç",)))
    print("Danışman (OpenAI seçili)  ", run_page("Yapay Zekâ Satış Danışmanı", state={"_ai_provider": "openai"}))
    print("Müşteri -> Arama geçişi   ", run_page("Müşteri DNA", buttons=("Bu ürün için Sıradaki 10 Arama",)))
