"""
yt_audio_collector.py — Engineering Audio Collector
Fulfills the "Hybrid-Dataset Summariser" specification.
Downloads high-fidelity audio for local OpenAI Whisper transcription.

Run from repo root:
    python src/data/yt_audio_collector.py
    python src/data/yt_audio_collector.py --target 1200
"""

import json
import logging
import time
import argparse
import shutil
import subprocess
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import yt_dlp
from tqdm import tqdm

# ── CONFIG (Fulfills Specification for Audio Collection) ───────────────
TARGET = 1200
OUTPUT_DIR = Path("data/raw/videos")
LOG_DIR = Path("logs")
COOKIE_PATH = Path(__file__).resolve().parent.parent.parent / "cookies.txt"

# Valid Video Filters
MIN_DUR = 5 * 60        # 5 minutes
MAX_DUR = 90 * 60       # 90 minutes
MIN_VIEWS = 1000
MAX_PER_CHANNEL = 10

# ── DOMAIN ALIGNMENT (Matches ArXiv [A1] Categories) ─────────────────────────
QUERIES = {
    "cs.AI": [
        "Artificial Intelligence university lecture",
        "knowledge representation AI",
        "AI planning search algorithms lecture",
        "intelligent agents multiagent systems lecture",
        "Artificial Intelligence full course 2024",
        "reasoning under uncertainty AI lecture",
    ],
    "cs.LG": [
        "Machine Learning theory lecture",
        "Deep Learning backpropagation",
        "reinforcement learning policy gradient lecture",
        "neural network optimization lecture university",
        "machine learning full course beginner",
        "support vector machine kernel lecture",
        "generative models VAE GAN lecture",
    ],
    "cs.CL": [
        "Natural Language Processing transformer lecture",
        "NLP university lecture",
        "attention mechanism transformer explained",
        "word embeddings word2vec lecture",
        "language model BERT GPT lecture",
        "text classification sentiment analysis lecture",
    ],
    "cs.CV": [
        "Computer Vision CNN lecture",
        "Image processing university",
        "object detection YOLO lecture explained",
        "image segmentation deep learning lecture",
        "convolutional neural network architecture lecture",
        "visual recognition feature extraction lecture",
    ],
    "cs.RO": [
        "Robotics kinematics lecture",
        "Autonomous systems SLAM lecture",
        "robot motion planning lecture university",
        "control systems PID lecture",
        "robot operating system ROS tutorial lecture",
        "robotic perception sensors lecture",
    ],
    "cs.SE": [
        "Software Engineering architecture lecture",
        "Design patterns university",
        "software testing methodology lecture",
        "agile scrum software development lecture",
        "system design interview lecture",
        "software requirements engineering lecture",
    ],
    "cs.DS": [
        "Data Structures algorithms university",
        "Graph theory lecture university",
        "dynamic programming algorithms lecture",
        "sorting algorithms analysis lecture",
        "binary tree heap priority queue lecture",
        "algorithm complexity Big O lecture",
    ],
}

AUDIO_EXTS = {".mp4", ".webm", ".mkv", ".m4a", ".opus", ".mp3", ".aac",
              ".ogg", ".flv", ".ts", ".wav", ".weba"}

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "audio_collector.log")],
)
log = logging.getLogger(__name__)


# ── ROBUST DOWNLOADER [A2] ────────────────────────────────────────────────────
def download_robust_audio(video_id: str, out_dir: Path) -> Optional[Path]:
    vid_dir = out_dir / video_id
    vid_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": "best",
        "outtmpl": str(vid_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
        "socket_timeout": 30,
        "extractor_args": {"youtube": {"player_client": ["android", "web", "tv_embedded"]}},
    }

    if COOKIE_PATH.exists():
        opts["cookiefile"] = str(COOKIE_PATH)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=True)
            dl_file = Path(ydl.prepare_filename(info))

            # prepare_filename can predict wrong extension — scan directory as fallback
            if not dl_file.exists():
                candidates = [f for f in vid_dir.iterdir()
                              if f.suffix.lower() in AUDIO_EXTS and f.name != "audio.wav"]
                if not candidates:
                    raise FileNotFoundError("No downloaded media file found")
                dl_file = candidates[0]

            # Convert to 16kHz mono WAV for Whisper
            wav_path = vid_dir / "audio.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(dl_file),
                "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", str(wav_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            dl_file.unlink(missing_ok=True)
            return wav_path

    except Exception as e:
        log.warning(f"Download failed [{video_id}]: {e}")
        shutil.rmtree(vid_dir, ignore_errors=True)
        return None


# ── MAIN PROCESSOR ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if COOKIE_PATH.exists():
        log.info(f"Using cookies from {COOKIE_PATH}")
    else:
        log.warning("cookies.txt not found — YouTube will rate-limit around ~300 videos")

    # Resume: rebuild done_ids AND channel_counts from existing meta.json files
    done_ids = set()
    channel_counts = defaultdict(int)
    for meta_path in OUTPUT_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            done_ids.add(meta_path.parent.name)
            channel_counts[meta.get("channel_id", "")] += 1
        except Exception:
            pass

    log.info(f"Resuming: {len(done_ids)} done, need {args.target - len(done_ids)} more")

    if len(done_ids) >= args.target:
        log.info("Target already reached!")
        return

    pbar = tqdm(total=args.target, initial=len(done_ids), unit="video")
    all_queries = [(cat, q) for cat, qs in QUERIES.items() for q in qs]

    consecutive_failures = 0

    while len(done_ids) < args.target:
        random.shuffle(all_queries)
        cycle_found = 0

        for cat, query in all_queries:
            if len(done_ids) >= args.target:
                break

            log.info(f"Searching [{cat}]: {query}")
            search_opts = {"extract_flat": True, "quiet": True, "no_warnings": True}
            if COOKIE_PATH.exists():
                search_opts["cookiefile"] = str(COOKIE_PATH)

            try:
                with yt_dlp.YoutubeDL(search_opts) as ydl:
                    results = ydl.extract_info(
                        f"ytsearch50:{query}", download=False
                    ).get("entries", [])
            except Exception as e:
                log.warning(f"Search failed [{query}]: {e}")
                time.sleep(5)
                continue

            for entry in (results or []):
                if len(done_ids) >= args.target:
                    break

                vid_id = entry.get("id")
                if not vid_id or vid_id in done_ids:
                    continue

                # Filters (unchanged from working Gemini script)
                if entry.get("live_status") == "is_live":
                    continue
                channel_id = entry.get("channel_id", "")
                if channel_counts[channel_id] >= MAX_PER_CHANNEL:
                    continue
                if (entry.get("view_count") or 0) < MIN_VIEWS:
                    continue
                duration = entry.get("duration") or 0
                if not (MIN_DUR <= duration <= MAX_DUR):
                    continue

                log.info(f"  Downloading {vid_id} ({duration // 60}m) [{cat}]")
                wav = download_robust_audio(vid_id, OUTPUT_DIR)

                if wav:
                    meta = {
                        "id": vid_id,
                        "title": entry.get("title", ""),
                        "channel_id": channel_id,
                        "duration_seconds": duration,
                        "domain": "engineering",
                        "modality": "video",
                        "category": cat,
                    }
                    (OUTPUT_DIR / vid_id / "meta.json").write_text(
                        json.dumps(meta, indent=2)
                    )
                    done_ids.add(vid_id)
                    channel_counts[channel_id] += 1
                    cycle_found += 1
                    consecutive_failures = 0
                    pbar.update(1)
                    pbar.set_postfix(cat=cat[3:], total=len(done_ids))
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 25:
                        pbar.close()
                        log.error(
                            "25 consecutive failures — stopping cleanly.\n"
                            "Likely fix: re-export cookies.txt from browser, "
                            "then re-run. Script will resume from where it stopped."
                        )
                        log.info(f"Saved so far: {len(done_ids)} videos")
                        sys.exit(1)

            time.sleep(1.5)

        if cycle_found == 0:
            log.warning("Full query cycle yielded no new videos — search space exhausted")
            break

    pbar.close()
    log.info(f"Done. Total: {len(done_ids)} videos")


if __name__ == "__main__":
    main()