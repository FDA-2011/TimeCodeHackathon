"""
kb_checker.py — определяет, относится ли кластер к кибербезопасности.

Всё локально, без API и без ручной установки моделей.
При первом запуске модель Qwen2.5 автоматически скачивается
с Hugging Face (~3 ГБ для 1.5B) и кэшируется в ~/.cache/huggingface.
Дальше работает офлайн.

Пайплайн:
    1. Keyword-фильтр (обязательное условие).
    2. LLM Qwen2.5-Instruct — отвечает "да" или "нет".
    3. Если модель не загрузилась — fallback на keyword.
"""

import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import warnings

warnings.filterwarnings("ignore")


# Keyword-фильтр
KB_KEYWORDS = [
    "уязвим", "взлом", "взломали", "хакер", "хакерск",
    "утечк", "утечка данных", "утечка информац",
    "ddos", "ддос", "фишинг", "шифровальщик", "вымогател",
    "кибератак", "кибербезопасн", "киберпреступ",
    "инцидент иб", "информационн безопасн",
    "пентест", "malware", "эксплойт", "backdoor",
    "компрометац", "троян", "ботнет", "sql-инъекц", "xss",
    "zero-day", "0day", "cve-", "антивирус",
    "защит данн", "аутентификац", "несанкционирован доступ",
    "похитили данн", "похищен данн", "взлом аккаунт",
]


def _cluster_text(cluster, max_chars: int = 1500) -> str:
    parts = []
    for it in cluster:
        if isinstance(it, dict):
            title = it.get("title") or ""
            text = it.get("text") or ""
        else:
            title = getattr(it, "title", "") or ""
            text = getattr(it, "text", "") or ""
        parts.append(f"{title.strip()}. {text[:400].strip()}")
    return " ".join(parts)[:max_chars]


def has_kb_keyword(cluster) -> bool:
    text = _cluster_text(cluster).lower()
    hits = [w for w in KB_KEYWORDS if w in text]
    if hits:
        print(f"[kb_checker] keyword hits: {hits}", flush=True)
    return bool(hits)



# LLM (Qwen2.5-Instruct, скачивается автоматически)
_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

_pipe = None
_load_failed = False


def _get_pipe():
    """
    Ленивая загрузка. Первый вызов скачивает модель с HF (~3 ГБ),
    дальше берётся из кэша.
    """
    global _pipe, _load_failed
    if _pipe is not None:
        return _pipe
    if _load_failed:
        return None

    try:
        from transformers import pipeline

        print(f"[kb_checker] Загружаю {_MODEL_NAME}...", flush=True)
        print("[kb_checker] При первом запуске модель скачается "
              "с Hugging Face (~3 ГБ, займёт несколько минут).",
              flush=True)

        _pipe = pipeline(
            "text-generation",
            model=_MODEL_NAME,
            device=-1,  # CPU
            torch_dtype="auto",
        )
        print("[kb_checker] Модель готова.", flush=True)
        return _pipe
    except Exception as e:
        print(f"[kb_checker] Не удалось загрузить модель: {e}", flush=True)
        _load_failed = True
        return None


_PROMPT = """Ты — классификатор новостей по теме «кибербезопасность».
Отвечай ТОЛЬКО одним словом: да или нет.

да — если новость про: хакерские атаки, взломы, утечки данных,
уязвимости, DDoS, фишинг, шифровальщики, киберпреступность,
инциденты информационной безопасности, защиту информации.

нет — если новость про: политику, экономику, спорт, культуру,
происшествия без ИТ-контекста, военные сводки, космос, образование.

нет — если упоминается «безопасность» в другом смысле:
Совет безопасности, национальная безопасность, продовольственная
безопасность, пожарная безопасность, общественная безопасность.

нет — если это политическое заявление, в котором слово
«кибербезопасность» используется как риторика, но само событие
не связано с ИТ-инцидентом или защитой данных.

Новость:
{text}

Ответ:"""


def _parse_yes_no(ans: str) -> bool | None:
    if not ans:
        return None
    ans = ans.strip().lower()
    head = ans[:30]
    # "нет" содержит "не", поэтому проверяем первым
    if re.search(r"\bнет\b", head):
        return False
    if re.search(r"\bда\b", head):
        return True
    return None


def _llm_verdict(text: str) -> bool | None:
    pipe = _get_pipe()
    if pipe is None:
        return None

    try:
        messages = [
            {"role": "system",
             "content": "Отвечай строго одним словом: да или нет."},
            {"role": "user", "content": _PROMPT.format(text=text)},
        ]
        out = pipe(
            messages,
            max_new_tokens=5,
            do_sample=False,
            return_full_text=False,
        )
        ans = out[0]["generated_text"]
        return _parse_yes_no(ans)
    except Exception as e:
        print(f"[kb_checker] Ошибка инференса: {e}", flush=True)
        return None


# Публичный API
def kb_probability(cluster, use_llm: bool = True) -> float:
    """
    Возвращает:
        1.0 — КБ (LLM сказал "да")
        0.0 — не КБ (LLM сказал "нет" или нет КБ-слов)
        0.7 — LLM недоступна, сработал keyword-фильтр
    """
    if not cluster:
        return 0.0

    if not has_kb_keyword(cluster):
        return 0.0

    if not use_llm:
        return 0.7

    text = _cluster_text(cluster)
    verdict = _llm_verdict(text)
    if verdict is None:
        return 0.7
    return 1.0 if verdict else 0.0


def is_kb_cluster(cluster, threshold: float = 0.5,
                  use_llm: bool = True) -> bool:
    return kb_probability(cluster, use_llm) >= threshold


def classify_clusters(clusters, threshold: float = 0.5,
                      use_llm: bool = True):
    """
    Батч-обработка. LLM вызывается только для кластеров,
    прошедших keyword-фильтр.
    """
    if not clusters:
        return []

    out = []
    for c in clusters:
        prob = kb_probability(c, use_llm)
        out.append((c, prob, prob >= threshold))
    return out
