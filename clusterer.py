from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


@dataclass
class RawNewsItem:
    id: str
    title: str
    text: str
    url: str
    pub_date: str
    source_name: str
    source_type: str


# Загружаем модель один раз при импорте
MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def embed(items: list[RawNewsItem]) -> np.ndarray:
    #Считает нормализованные эмбеддинги для списка новостей.
    texts = [f"{it.title}. {it.text}".strip() for it in items]
    return MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def same_event(a: RawNewsItem, b: RawNewsItem, threshold: float = 0.75) -> bool:
    #Об одном ли событии две новости.
    vecs = embed([a, b])
    score = float(vecs[0] @ vecs[1])
    return score >= threshold


def cluster(items: list[RawNewsItem], threshold: float = 0.75) -> list[list[RawNewsItem]]:
    #Группирует новости в кластеры. Каждый кластер — одно событие.
    if len(items) < 2:
        return [items] if items else []

    vecs = embed(items)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(vecs)

    groups: dict[int, list[RawNewsItem]] = {}
    for item, label in zip(items, labels):
        groups.setdefault(int(label), []).append(item)

    # Большие кластеры — вперёд
    return sorted(groups.values(), key=len, reverse=True)
