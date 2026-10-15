from parser import parse_sources
# тест парсера


SOURCES = [
    {"name": "Хакер", "url": "https://xakep.ru/feed/", "type": "rss"},
    {"name": "Инфобезопасность", "url": "https://infobezopasnost.ru/blog/news/feed/", "type": "rss"},
    {"name": "ComNews IB", "url": "https://www.comnews.ru/rss/information_security", "type": "rss"},
    {"name": "ТАСС Общий", "url": "https://tass.ru/rss/v2.xml", "type": "rss"},
]

def main():
    print("🚀 Запуск теста парсера...\n")

    news, journal = parse_sources(SOURCES)

    print(f"📦 Новостей собрано: {len(news)}")
    print(f"📋 Записей в журнале: {len(journal)}\n")

    print("--- ЖУРНАЛ ---")
    for entry in journal:
        status_icon = "✅" if entry["status"] == "ok" else "❌"
        print(f"{status_icon} [{entry['source']}] {entry['status']} | "
              f"Записей: {entry['items_found']} | {entry.get('reason', '')}")

    if news:
        print("\n--- ОБРАЗЕЦ НОВОСТИ ---")
        n = news[0]
        print(f"ID:       {n.id}")
        print(f"Заголовок:{n.title}")
        print(f"Дата:     {n.pub_date}")
        print(f"Источник: {n.source_name}")
        print(f"URL:      {n.url}")
        print(f"Текст:    {n.text[:150]}...")
    else:
        print("\n⚠️ Новостей нет. Проверь журнал выше на наличие ошибок.")


if __name__ == "__main__":
    main()
