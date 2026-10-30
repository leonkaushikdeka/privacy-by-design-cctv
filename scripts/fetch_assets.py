"""Download model weights and openly licensed sample images used by the notebook."""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"
UA = {"User-Agent": "privacy-cctv-student-project/1.0 (educational use)"}

MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
    "efficientdet_lite2.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite",
}


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def commons_search(query, limit):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1024,
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        pages = json.load(r).get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        info = p["imageinfo"][0]
        meta = info.get("extmetadata", {})
        out.append({
            "title": p["title"],
            "thumb": info["thumburl"],
            "page": info["descriptionurl"],
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "artist": meta.get("Artist", {}).get("value", ""),
        })
    return out


def fetch_samples(queries, limit):
    SAMPLES.mkdir(parents=True, exist_ok=True)
    index_path = SAMPLES / "sources.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for tag, query in queries.items():
        for i, item in enumerate(commons_search(query, limit)):
            name = f"{tag}_{i:02d}.jpg"
            try:
                download(item["thumb"], SAMPLES / name)
            except Exception as e:
                print("skip", item["title"], e)
                continue
            index[name] = {**item, "query": query}
            time.sleep(0.3)
    index_path.write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        download(url, MODELS / name)
        print("model", name, (MODELS / name).stat().st_size)
    if "--samples" in sys.argv:
        fetch_samples({
            "pos": "card payment terminal PIN hand",
            "upi": "UPI payment smartphone",
            "phone": "hands typing smartphone",
            "people": "people walking corridor",
        }, limit=8)
