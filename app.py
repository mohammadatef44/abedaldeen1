from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

import requests
from flask import Flask, render_template, request
from PIL import Image

from ml.model import blend_scores, load_model, predict_text

BASE_DIR = Path(__file__).resolve().parent
INSIGHTS_PATH = BASE_DIR / "data" / "insights.json"

app = Flask(__name__)
model = load_model()


SUSPICIOUS_KEYWORDS = {"guaranteed", "miracle", "secret", "instant", "shocking", "alien"}
TRUSTED_DOMAINS = {".gov", ".edu", "reuters.com", "apnews.com", "bbc.com"}


def analyze_image(image_url: str) -> Tuple[float, Dict[str, str]]:
    if not image_url:
        return 0.5, {"note": "No image URL provided."}
    try:
        response = requests.get(image_url, timeout=4)
        response.raise_for_status()
        image = Image.open(response.raw)
        width, height = image.size
        palette = image.resize((1, 1)).getpixel((0, 0))
        brightness = sum(palette) / (3 * 255)
        score = 0.4 if brightness > 0.6 else 0.6
        details = {
            "resolution": f"{width}x{height}",
            "dominant_color": f"rgb{palette}",
            "brightness": f"{brightness:.2f}",
        }
        return score, details
    except Exception as exc:  # noqa: BLE001 - user-facing insight
        return 0.55, {"note": f"Unable to analyze image: {exc}"}


def analyze_video(video_url: str) -> Tuple[float, Dict[str, str]]:
    if not video_url:
        return 0.5, {"note": "No video URL provided."}
    lower = video_url.lower()
    score = 0.5
    hints = []
    if any(ext in lower for ext in (".mp4", ".mov", ".webm")):
        score = 0.45
        hints.append("Direct video file detected.")
    if "live" in lower:
        score = 0.55
        hints.append("Contains 'live' keyword.")
    return score, {"signals": ", ".join(hints) or "Standard video URL."}


def analyze_link(link_url: str) -> Tuple[float, Dict[str, str]]:
    if not link_url:
        return 0.5, {"note": "No source link provided."}
    parsed = urlparse(link_url)
    hostname = parsed.netloc.lower()
    score = 0.55
    trust = "Unknown source"
    if any(domain in hostname for domain in TRUSTED_DOMAINS):
        score = 0.35
        trust = "Trusted domain"
    if hostname.endswith(".blog") or hostname.endswith(".xyz"):
        score = 0.65
        trust = "Low reputation TLD"
    return score, {"source": hostname or "Unknown", "trust": trust}


def keyword_risk(text: str) -> float:
    words = set(text.lower().split())
    hits = words.intersection(SUSPICIOUS_KEYWORDS)
    if not hits:
        return 0.4
    return min(0.8, 0.45 + 0.1 * len(hits))


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        title = request.form.get("title", "")
        body = request.form.get("body", "")
        image_url = request.form.get("image_url", "")
        video_url = request.form.get("video_url", "")
        link_url = request.form.get("link_url", "")

        text_result = predict_text(model, title, body)
        text_score = text_result.probability_fake
        keyword_score = keyword_risk(f"{title} {body}")
        media_score, image_details = analyze_image(image_url)
        video_score, video_details = analyze_video(video_url)
        link_score, link_details = analyze_link(link_url)

        combined_media = (media_score + video_score) / 2
        combined_text = (text_score + keyword_score) / 2
        final_score, label = blend_scores(combined_text, combined_media, link_score)

        result = {
            "label": label,
            "final_score": f"{final_score:.2f}",
            "text_label": text_result.label,
            "text_fake": f"{text_result.probability_fake:.2f}",
            "text_real": f"{text_result.probability_real:.2f}",
            "image": image_details,
            "video": video_details,
            "link": link_details,
        }

    insights = {}
    if INSIGHTS_PATH.exists():
        insights = json.loads(INSIGHTS_PATH.read_text())

    return render_template("index.html", result=result, insights=insights)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
