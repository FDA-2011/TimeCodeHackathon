import feedparser
import requests
import re
import logging
from dataclasses import dataclass

# подключаем библиотека
logger = logging.getLogger(__name__)  # логи


@dataclass
class RawNewsItem:
    id: str
    title: str
    text: str
    url: str
    pub_date: str
    source_name: str
    source_type: str


# объект новости
def strip_html(text):
    clean = re.sub(r'<[^>]+>', ' ', text)  # страшное регулярное выражение которое очищает статью от html тегов
    return re.sub(r'\s+', ' ', clean).strip()  # убираем лишние пробелы


def parse_rss(url, name):
    journal_entry = {"source": name, "url": url, "status": "ok", "items_found": 0, "reason": ""}  # данные о парсинге
    items = []

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "MyBot/1.0"})  # запрос к сайту
        resp.raise_for_status()  # возвращает ошибку если не получилось посмотреть страницу

        feed = feedparser.parse(resp.text)  # парсим текст

        for entry in feed.entries:  # смотри все новости
            try:
                item_id = entry.get("id") or entry.get("link") or entry.get("title", "")
                pub_date = entry.get("published") or entry.get("updated") or ""
                raw_text = entry.get("summary") or entry.get("description") or ""

                items.append(RawNewsItem(
                    id=str(item_id),
                    title=entry.get("title", "Без заголовка").strip(),
                    text=strip_html(raw_text),
                    url=entry.get("link", ""),
                    pub_date=str(pub_date),
                    source_name=name,
                    source_type="rss"
                ))  # добавляем новость
            except Exception as e:
                logger.warning(f"[{name}] Ошибка: {e}")  # логируем ошибку

        journal_entry["items_found"] = len(items)

    except Exception as e:
        journal_entry["status"] = "error"
        journal_entry["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        logger.error(f"[{name}] Failed: {journal_entry['reason']}")

    return items, journal_entry


def parse_sources(sources):
    all_news = []
    journal = []

    for src in sources:  # смотрим все ресуры
        if src.get("type", "rss") == "rss":
            news, entry = parse_rss(src["url"], src["name"])  # rss парсер
            all_news.extend(news)
            journal.append(entry)
        else:
            journal.append({
                "source": src["name"], "url": src.get("url", ""),
                "status": "skipped", "items_found": 0,
                "reason": "html_not_implemented_yet"
            })  # Иначе пропускаем

    return all_news, journal
