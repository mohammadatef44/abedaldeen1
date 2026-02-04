from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "news_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "news_model.joblib"


@dataclass
class ModelResult:
    label: str
    probability_fake: float
    probability_real: float


def train_model() -> Pipeline:
    dataset = pd.read_csv(DATA_PATH)
    dataset["text"] = dataset["title"].fillna("") + " " + dataset["body"].fillna("")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(dataset["text"], dataset["label"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


def predict_text(model: Pipeline, title: str, body: str) -> ModelResult:
    text = f"{title} {body}".strip()
    if not text:
        return ModelResult(label="unknown", probability_fake=0.5, probability_real=0.5)
    proba = model.predict_proba([text])[0]
    classes = model.classes_.tolist()
    score_map: Dict[str, float] = dict(zip(classes, proba))
    prob_fake = score_map.get("fake", 0.5)
    prob_real = score_map.get("real", 0.5)
    label = "fake" if prob_fake >= prob_real else "real"
    return ModelResult(label=label, probability_fake=prob_fake, probability_real=prob_real)


def blend_scores(text_score: float, media_score: float, link_score: float) -> Tuple[float, str]:
    combined = 0.6 * text_score + 0.25 * media_score + 0.15 * link_score
    label = "Likely Real" if combined < 0.5 else "Likely Fake"
    return combined, label
