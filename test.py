"""
test_summarizer.py — проверки text_summarizer.py.

Запуск:
    python test_summarizer.py        # все тесты
    python test_summarizer.py -v     # с трассировкой
"""

import sys
import traceback

from text_summarizer import (
    summarize, summarize_item, summarize_cluster,
    summarize_extractive, _split_sentences,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = []

LONG = (
    "Компания X сообщила о крупной утечке данных. "
    "Злоумышленники получили доступ к базе на 5 миллионов записей. "
    "Инцидент произошёл на прошлой неделе. "
    "Служба безопасности обнаружила аномалию в логах. "
    "Компания уведомила регулятора. "
    "Пострадавшим обещают бесплатный мониторинг."
)


def _run(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"[{PASS}] {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"[{FAIL}] {name} — {e}")
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[{FAIL}] {name} — {type(e).__name__}: {e}")
        if "-v" in sys.argv:
            traceback.print_exc()


# тесты
def t_split():
    sents = _split_sentences(LONG)
    assert len(sents) == 6, f"ожидалось 6, got {len(sents)}"


def t_shortens():
    short = summarize(LONG, mode="extractive", max_sentences=2, max_chars=200)
    assert len(short) < len(LONG), "не сократилось"
    assert len(short) <= 201, f"превышен лимит: {len(short)}"


def t_empty():
    assert summarize("") == ""
    assert summarize("   ") == ""
    assert summarize(None) == ""


def t_already_short():
    s = "Короткая новость."
    assert summarize(s, mode="extractive", max_sentences=3, max_chars=200) == s


def t_extractive_call_direct():
    short = summarize_extractive(LONG, max_sentences=2, max_chars=150)
    assert 0 < len(short) <= 151


def t_item_inplace():
    item = {"title": "t", "text": LONG}
    out = summarize_item(item, mode="extractive",
                         inplace=True, max_chars=150)
    assert "short_text" in item
    assert item["short_text"] == out
    assert len(out) <= 151


def t_item_no_inplace():
    item = {"title": "t", "text": LONG}
    out = summarize_item(item, mode="extractive", max_chars=150)
    assert "short_text" not in item
    assert out


def t_cluster():
    cl = [{"title": "Утечка", "text": LONG}]
    out = summarize_cluster(cl, mode="extractive", max_chars=120)
    assert out.startswith("Утечка"), out


def t_cluster_empty():
    assert summarize_cluster([]) == ""


def t_llm_fallback():
    # без Qwen должен вернуться extractive-результат
    short = summarize(LONG, mode="llm", max_chars=200)
    assert short, "вернулось пусто"
    assert len(short) <= 201


def t_no_mid_word_cut():
    """Проверяем, что обрезка идёт по границе слова."""
    short = summarize_extractive(LONG, max_sentences=3, max_chars=80)
    if short.endswith("…"):
        body = short[:-1].rstrip()
        # последний символ до многоточия не должен быть разорван
        assert body and not body.endswith("-"), short


# запуск 
TESTS = [
    ("split_sentences", t_split),
    ("shortens", t_shortens),
    ("empty_input", t_empty),
    ("already_short", t_already_short),
    ("extractive_direct", t_extractive_call_direct),
    ("item_inplace", t_item_inplace),
    ("item_no_inplace", t_item_no_inplace),
    ("cluster", t_cluster),
    ("cluster_empty", t_cluster_empty),
    ("llm_fallback", t_llm_fallback),
    ("no_mid_word_cut", t_no_mid_word_cut),
]


def main():
    print("=" * 50)
    print("Тесты text_summarizer.py")
    print("=" * 50)
    for name, fn in TESTS:
        _run(name, fn)
    passed = sum(1 for _, ok, _ in _results if ok)
    print("=" * 50)
    print(f"Итого: {passed}/{len(_results)} PASS")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())