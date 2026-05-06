import os
import subprocess
import cv2
import numpy as np
import whisper
import av
import torch
from fastapi import FastAPI, UploadFile, File
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

app = FastAPI()

# -----------------------------
# MODELLER (startup'ta yükle)
# -----------------------------
whisper_model = whisper.load_model("base")

processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# -----------------------------
# UTILS
# -----------------------------
def extract_audio(video_path, audio_path="audio.wav"):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            time_sec = frame_id / video_fps
            frames.append(frame)
            timestamps.append(time_sec)

        frame_id += 1

    cap.release()
    return frames, timestamps


def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["segments"]


def align(frame_times, segments):
    aligned = []
    for t in frame_times:
        best = None
        min_dist = 1e9

        for seg in segments:
            center = (seg["start"] + seg["end"]) / 2
            dist = abs(center - t)
            if dist < min_dist:
                min_dist = dist
                best = seg

        aligned.append({"time": t, "text": best["text"] if best else ""})
    return aligned


def build_context(aligned):
    return "\n".join([
        f"{a['time']:.1f}s: {a['text']}"
        for a in aligned if a["text"].strip()
    ])


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)

    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# -----------------------------
# ANALYSIS CORE
# -----------------------------
def analyze(video_path):
    audio_path = extract_audio(video_path)
    frames, timestamps = extract_frames(video_path)
    segments = transcribe_audio(audio_path)
    aligned = align(timestamps, segments)

    context_text = build_context(aligned)

    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, 8).astype(int)
    clip = read_video_pyav(container, indices)

    prompt = f"""
USER: <video>

Transcript:
{context_text}

Return JSON:
{{
  "summary": "...",
  "tags": ["..."],
  "intent": "..."
}}

ASSISTANT:
"""

    inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_length=200)

    raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return raw


# -----------------------------
# API
# -----------------------------
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    video_path = f"/tmp/{file.filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    result = analyze(video_path)

    return {"result": result}