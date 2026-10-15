"""
text_summarizer.py — сокращает текст новости.

Два режима:
    extractive — быстро, без моделей (TF-оценка предложений)
    llm        — через Qwen, если он уже загружен в kb_checker,
                 иначе автоматический fallback на extractive

Работает и с dict, и с RawNewsItem.
"""

from __future__ import annotations

import re
from collections import Counter

# Доступ к полям (dict / dataclass)
def _get(obj, key: str, default: str = "") -> str:
    v = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    return "" if v is None else str(v)


# Утилиты
_SENT = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9\-]{3,}")

_STOP = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне",
    "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему",
    "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже",
    "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего",
    "ей", "может", "они", "тут", "где", "есть", "надо", "ней",
    "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два",
    "об", "другой", "хоть", "после", "над", "больше", "тот",
    "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
    "нельзя", "такой", "им", "более", "всегда", "конечно",
    "всю", "между", "это", "этих", "эта", "этими", "эти",
}


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


# Экстрактивный режим
def _score_sentences(sents: list[str]) -> list[float]:
    if not sents:
        return []

    freq: Counter[str] = Counter()
    for s in sents:
        for w in _tokens(s):
            if w not in _STOP:
                freq[w] += 1
    if not freq:
        return [0.0] * len(sents)

    max_f = max(freq.values())
    out = []
    for i, s in enumerate(sents):
        words = [w for w in _tokens(s) if w not in _STOP]
        if not words:
            out.append(0.0)
            continue
        base = sum(freq[w] for w in words) / (len(words) * max_f)
        pos = 1.0 if i == 0 else (0.9 if i == 1 else 0.8)
        n = len(words)
        pen = 0.6 if n < 4 else (0.85 if n > 40 else 1.0)
        out.append(base * pos * pen)
    return out


def summarize_extractive(text: str, max_sentences: int = 2,
                         max_chars: int = 300) -> str:
    """Топ-N предложений по TF-оценке."""
    sents = _split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        out = " ".join(sents)
        return out if len(out) <= max_chars else out[:max_chars].rsplit(" ", 1)[0] + "…"

    scores = _score_sentences(sents)
    top = sorted(sorted(range(len(sents)),
                        key=lambda i: scores[i], reverse=True)[:max_sentences])
    out = " ".join(sents[i] for i in top)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


# LLM-режим (переиспользует загрузчик Qwen из kb_checker)
_PROMPT = """Сократи новость до 1–2 коротких предложений.
Сохрани суть: кто, что сделал, где, когда.
Без вступлений и без «в этой новости говорится».
Пиши на русском. Ответ — только сжатый текст.

Новость:
{text}

Сокращённо:"""


def _llm_shorten(text: str, max_chars: int = 300) -> str | None:
    try:
        from kb_checker import _get_pipe
    except Exception:
        return None
    pipe = _get_pipe()
    if pipe is None:
        return None
    try:
        out = pipe(
            [{"role": "system", "content": "Ты кратко пересказываешь новости."},
             {"role": "user", "content": _PROMPT.format(text=text[:2000])}],
            max_new_tokens=120, do_sample=False, return_full_text=False,
        )
        ans = out[0]["generated_text"].strip()
        ans = re.split(r"\n\s*Новость|\n\s*Ответ", ans)[0].strip()
        if not ans:
            return None
        if len(ans) > max_chars:
            ans = ans[:max_chars].rsplit(" ", 1)[0] + "…"
        return ans
    except Exception:
        return None


# Публичный API
def summarize(text: str, mode: str = "extractive",
              max_sentences: int = 2, max_chars: int = 300) -> str:
    """
    mode="extractive" — быстро, без моделей
    mode="llm"        — через Qwen, с fallback на extractive
    """
    text = (text or "").strip()
    if not text:
        return ""
    if mode == "llm":
        r = _llm_shorten(text, max_chars)
        if r:
            return r
    return summarize_extractive(text, max_sentences, max_chars)


def summarize_item(item, mode: str = "extractive",
                   max_sentences: int = 2, max_chars: int = 300,
                   inplace: bool = False) -> str:
    """Сокращает поле text; при inplace=True пишет в short_text."""
    short = summarize(_get(item, "text"), mode=mode,
                      max_sentences=max_sentences, max_chars=max_chars)
    if inplace and isinstance(item, dict):
        item["short_text"] = short
    elif inplace:
        try:
            setattr(item, "short_text", short)
        except Exception:
            pass
    return short


def summarize_cluster(cluster, mode: str = "extractive",
                      max_chars: int = 400) -> str:
    """Заголовок лидера + сжатый текст лидера."""
    if not cluster:
        return ""
    rep = cluster[0]
    title = _get(rep, "title").strip()
    body = summarize(_get(rep, "text"), mode=mode, max_chars=max_chars)
    if title and body:
        return f"{title}. {body}"
    return title or body


# CLI
if __name__ == "__main__":
    import sys
    sample = (
        "Компания X сообщила о крупной утечке данных. "
        "Злоумышленники получили доступ к базе клиентов на 5 миллионов записей. "
        "Инцидент произошёл на прошлой неделе и был обнаружен службой "
        "безопасности компании. Представители уже уведомили регулятора "
        "и начали внутреннее расследование. Пострадавшим обещают "
        "бесплатный мониторинг кредитной истории в течение года."
    )
    mode = sys.argv[1] if len(sys.argv) > 1 else "extractive"
    print(f"Режим: {mode}\n")
    print("Оригинал:\n" + sample)
    print(f"\n({len(sample)} символов)\n")
    short = summarize(sample, mode=mode, max_sentences=2, max_chars=250)
    print("Сокращённо:\n" + short)
    print(f"\n({len(short)} символов)")