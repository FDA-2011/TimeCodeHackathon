# app.py — Парсер + Фильтр + Кластеризация + КБ-фильтр
import warnings

warnings.filterwarnings("ignore")

import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
from datetime import datetime, timedelta, timezone


def log(msg):
    print(f">>> {msg}", flush=True, file=sys.stderr)


log("[1] start")

import pandas as pd
import requests
import feedparser
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from kb_checker import classify_clusters

log("[2] imports ok")

st.set_page_config(page_title="Код Времени", layout="wide")


# Модель эмбеддингов (кластеризация)
@st.cache_resource(show_spinner="Загружаю модель эмбеддингов...")
def get_embedder():
    log("[3] loading embedder")
    m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                            device="cpu")
    log("[4] embedder loaded")
    return m


EMBEDDER = get_embedder()


# Источники
RSS_SOURCES = [
    ("Инфобезопасность", "https://infobezopasnost.ru/blog/news/feed/"),
    ("ComNews IB", "https://www.comnews.ru/rss/information_security"),
    ("ТАСС Общий", "https://tass.ru/rss/v2.xml"),
    ("Хакер", "https://xakep.ru/feed/"),
]


def parse_rss(name, url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        items = []
        for e in feed.entries:
            items.append({
                "source_name": name,
                "title": getattr(e, "title", ""),
                "text": getattr(e, "summary", "")
                        or getattr(e, "description", ""),
                "url": getattr(e, "link", ""),
                "pub_date": getattr(e, "published", ""),
            })
        return items, None
    except Exception as ex:
        return [], f"{type(ex).__name__}: {ex}"


def load_news(test_mode: bool, uploaded_file):
    journal = []
    items = []

    if test_mode and uploaded_file is not None:
        data = json.load(uploaded_file)
        for row in data:
            items.append({
                "source_name": row.get("source_name", "?"),
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "url": row.get("url", ""),
                "pub_date": row.get("pub_date", ""),
            })
        journal.append({"source": "JSON", "url": "uploaded",
                        "status": "ok", "items_found": len(items),
                        "reason": ""})
        return items, journal

    for name, url in RSS_SOURCES:
        log(f"parsing {name}")
        news, err = parse_rss(name, url)
        if err:
            journal.append({"source": name, "url": url, "status": "error",
                            "items_found": 0, "reason": err})
        else:
            journal.append({"source": name, "url": url, "status": "ok",
                            "items_found": len(news), "reason": ""})
            items.extend(news)
    return items, journal



# Кластеризация
def cluster_news(items, threshold=0.65):
    if len(items) < 2:
        return [[it] for it in items]

    texts = [f"{it['title']}. {it['text'][:400]}".strip() for it in items]
    vecs = EMBEDDER.encode(texts, normalize_embeddings=True,
                           convert_to_numpy=True, batch_size=32,
                           show_progress_bar=False)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(vecs)

    groups = {}
    for it, lbl in zip(items, labels):
        groups.setdefault(int(lbl), []).append(it)
    return sorted(groups.values(), key=len, reverse=True)



# UI
st.title("🔍 Код Времени: Парсер, фильтр, кластеризация и КБ-фильтр")

with st.sidebar:
    st.header("⚙️ Настройки")

    test_mode = st.checkbox("🧪 Тестовый режим (загрузить JSON)")
    uploaded = None
    if test_mode:
        uploaded = st.file_uploader("Выбери JSON", type=["json"])

    st.divider()
    st.subheader("Параметры кластеризации")
    threshold = st.slider("Порог сходства", 0.50, 0.95, 0.65, 0.05)

    st.divider()
    st.subheader("Параметры КБ")
    kb_threshold = st.slider(
        "Порог КБ", 0.40, 0.90, 0.60, 0.05,
        help="Вероятность, выше которой кластер считается КБ",
    )
    use_llm = st.checkbox(
        "🤖 Использовать Qwen2.5 для КБ-фильтра",
        value=True,
        help="При первом запуске модель скачается автоматически (~3 ГБ). "
             "Без неё работает keyword-фильтр.",
    )

    st.divider()
    st.subheader("📅 Период отбора")
    period_mode = st.radio("Режим", ["Произвольный период",
                                     "24 часа до контрольного"])

    if period_mode == "Произвольный период":
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input(
                "Начало (дата)",
                value=datetime.now() - timedelta(days=1),
            )
            start_time = st.time_input("Начало (время)",
                                       value=datetime.now().time())
        with col_b:
            end_date = st.date_input("Конец (дата)", value=datetime.now())
            end_time = st.time_input("Конец (время)",
                                     value=datetime.now().time())
        start_dt = datetime.combine(start_date, start_time)
        end_dt = datetime.combine(end_date, end_time)
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(hours=24)

    st.caption(f"**Период:** {start_dt:%d.%m %H:%M} — {end_dt:%d.%m %H:%M}")

    st.divider()
    run = st.button("🚀 Запустить обработку", type="primary")

if run:
    log("[5] run clicked")

    with st.spinner("Парсинг источников..."):
        items, journal = load_news(test_mode, uploaded)
    log(f"[6] parsed {len(items)}")

    if not items:
        st.error("Не удалось загрузить новости.")
        st.stop()


    def in_period(it):
        s = it.get("pub_date", "")
        if not s:
            return True
        try:
            dt = pd.to_datetime(s, utc=True).to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (start_dt.replace(tzinfo=timezone.utc)
                    <= dt
                    <= end_dt.replace(tzinfo=timezone.utc))
        except Exception:
            return True


    filtered = [it for it in items if in_period(it)]
    log(f"[7] after date filter: {len(filtered)}")

    with st.spinner("Кластеризация..."):
        clusters = cluster_news(filtered, threshold=threshold)
    log(f"[8] clusters: {len(clusters)}")

    with st.spinner("Определяю КБ-кластеры..."):
        classified = classify_clusters(
            clusters,
            threshold=kb_threshold,
            use_llm=use_llm,
        )
    kb_count = sum(1 for _, _, is_kb in classified if is_kb)
    log(f"[9] kb count: {kb_count}")

    ok_sources = sum(1 for j in journal if j["status"] == "ok")
    err_sources = sum(1 for j in journal if j["status"] == "error")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Успешных источников", ok_sources)
    c2.metric("Ошибок источников", err_sources)
    c3.metric("Сырых новостей", len(items))
    c4.metric("После фильтра дат", len(filtered))
    c5.metric("Уникальных событий", len(clusters))
    c6.metric("Событий по КБ", kb_count)

    st.divider()

    tab_events, tab_journal, tab_raw, tab_tests = st.tabs(
        ["📰 События (кластеры)", "📋 Журнал обработки",
         "⚙️ Сырые данные", "🧪 Тесты"]
    )

    with tab_events:
        st.subheader(
            f"Найдено уникальных событий: {len(clusters)} "
            f"(Кибербезопасность: {kb_count})"
        )

        sorted_classified = sorted(
            classified, key=lambda t: (not t[2], -t[1])
        )

        only_kb = st.checkbox("Показать только КБ-события", value=False)
        shown = ([(c, p, k) for c, p, k in sorted_classified if k]
                 if only_kb else sorted_classified)

        for i, (cluster, prob, is_kb) in enumerate(shown, 1):
            head = f"**Событие #{i}** ({len(cluster)} источников)"
            if is_kb:
                head += f" | :green[🛡️ Кибербезопасность ({prob * 100:.1f}%)]"
            else:
                head += f" | :gray[📰 Общее ({(1 - prob) * 100:.1f}%)]"
            st.markdown(head)

            rep = cluster[0]
            st.markdown(rep["title"])
            if rep.get("text"):
                st.caption(rep["text"][:400])

            with st.expander("Показать все источники этого события"):
                for it in cluster:
                    st.caption(
                        f"[{it['source_name']}] {it.get('pub_date', '')} — "
                        f"[Link]({it.get('url', '')})"
                    )
            st.divider()

    with tab_journal:
        st.dataframe(pd.DataFrame(journal), use_container_width=True)

    with tab_raw:
        st.dataframe(pd.DataFrame(filtered), use_container_width=True)

    with tab_tests:
        st.info("Здесь будут результаты обязательных проверок из задания.")
