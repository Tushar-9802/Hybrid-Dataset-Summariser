"""
transcribe.py — Local Whisper Transcription [A2]
Transcribes validated audio WAVs using OpenAI Whisper (large-v3).

Optimized for RTX 5070 Ti (16GB VRAM):
  - FP16 inference (halves memory, faster compute)
  - Greedy decoding (temperature=0, fastest mode)
  - Pre-loaded audio arrays (reduces I/O bottleneck)

Expected speed: ~2-4 min per video (avg 29.7 min audio).

Prerequisites:
    pip install openai-whisper

Run from repo root:
    python src/data/transcribe.py
    python src/data/transcribe.py --model medium   # if OOM on large-v3
"""

import json
import argparse
import logging
import time
import sys
from pathlib import Path

import torch
import whisper

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw/videos")
LOG_DIR = Path("logs")

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "transcription.log"),
    ],
)
log = logging.getLogger(__name__)


def transcribe_all(model_name: str, language: str):
    # ── Device setup ──
    if not torch.cuda.is_available():
        log.error("CUDA not available. Check PyTorch installation.")
        log.error(f"torch version: {torch.__version__}")
        sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Device: {device} — {gpu_name} ({vram:.1f} GB)")

    # ── Load model ──
    log.info(f"Loading Whisper {model_name}...")
    model = whisper.load_model(model_name, device=device)
    log.info("Model loaded")

    # ── Collect work ──
    all_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "audio.wav").exists()
    ])

    already_done = [d for d in all_dirs if (d / "transcript.json").exists()]
    to_do = [d for d in all_dirs if not (d / "transcript.json").exists()]

    log.info(f"Total with audio: {len(all_dirs)}")
    log.info(f"Already transcribed: {len(already_done)}")
    log.info(f"Remaining: {len(to_do)}")

    if not to_do:
        log.info("Nothing to transcribe — all done.")
        return

    failed = []
    start_time = time.time()

    for i, vid_dir in enumerate(to_do):
        vid_id = vid_dir.name
        wav_path = vid_dir / "audio.wav"
        transcript_path = vid_dir / "transcript.json"

        elapsed = time.time() - start_time
        rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0
        eta_min = ((len(to_do) - i) / rate) if rate > 0 else 0

        log.info(
            f"[{i+1}/{len(to_do)}] {vid_id} "
            f"({rate:.1f} vids/min, ETA {eta_min:.0f}m)"
        )

        try:
            # Pre-load audio as numpy array — avoids repeated file I/O
            audio = whisper.load_audio(str(wav_path))

            result = model.transcribe(
                audio,
                language=language,
                task="transcribe",
                fp16=True,                           # FP16 on CUDA
                temperature=0,                       # Greedy decoding (fastest)
                condition_on_previous_text=False,     # Reduces hallucination
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

            transcript_data = {
                "id": vid_id,
                "text": result["text"].strip(),
                "word_count": len(result["text"].split()),
            }

            transcript_path.write_text(
                json.dumps(transcript_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        except torch.cuda.OutOfMemoryError:
            log.error(f"OOM on {vid_id} — clearing cache and skipping")
            torch.cuda.empty_cache()
            failed.append(vid_id)
            continue

        except Exception as e:
            log.error(f"Failed {vid_id}: {e}")
            failed.append(vid_id)
            continue

    total_time = (time.time() - start_time) / 60
    log.info(f"{'='*50}")
    log.info(f"Transcription complete")
    log.info(f"  Processed: {len(to_do) - len(failed)}/{len(to_do)}")
    log.info(f"  Failed:    {len(failed)}")
    log.info(f"  Time:      {total_time:.1f} minutes")
    if failed:
        log.info(f"  Failed IDs: {failed[:20]}")


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription for video audio")
    parser.add_argument("--model", default="large-v3",
                        choices=["small", "medium", "large-v3"],
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default="en",
                        help="Language code (default: en)")
    args = parser.parse_args()

    transcribe_all(args.model, args.language)


if __name__ == "__main__":
    main()