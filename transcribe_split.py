# Install requirements first:
# pip install torch torchaudio whisperx pyannote.audio tqdm

import torch
import whisperx
from pyannote.audio import Pipeline
from pathlib import Path
import json
import argparse
from tqdm import tqdm

# ------------------- Arguments -------------------
parser = argparse.ArgumentParser(description="Transcribe and chunk audio with speaker diarization")
parser.add_argument("audio_file", type=str, help="Path to the audio file")
parser.add_argument("--chunk_size", type=int, default=3, help="Number of transcript segments per chunk")
parser.add_argument("--full_transcript_json", type=str, default=None, help="Optional path to save full transcript JSON")
parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token with read access")
args = parser.parse_args()

audio_file = Path(args.audio_file)

# ------------------- Load Models -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Loading WhisperX ASR model...")
asr_model = whisperx.load_model("base", device=device, compute_type="float32")
print("✅ ASR model loaded.")

print("📥 Loading Pyannote speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=args.hf_token
)
print("✅ Pyannote pipeline loaded.")

# ------------------- Transcribe -------------------
print(f"🎤 Transcribing audio: {audio_file} ...")
result = asr_model.transcribe(str(audio_file))
segments = result["segments"]
for _ in tqdm(range(len(segments)), desc="Processing segments"):
    pass  # Just a visual placeholder for segment loop
print("✅ Transcription complete.")

# ------------------- Diarize -------------------
print("🎧 Running speaker diarization...")
diarization = pipeline(str(audio_file))
print("✅ Diarization complete.")

# Merge speaker info into transcript
print("🔗 Merging speaker info into transcript...")
for segment in tqdm(segments, desc="Assigning speakers"):
    start = segment["start"]
    end = segment["end"]
    speaker_label = "Unknown"
    for turn in diarization.itertracks(yield_label=True):
        track, _, speaker = turn
        if track.start <= start <= track.end:
            speaker_label = speaker
            break
    segment["speaker"] = speaker_label
print("✅ Speaker labels assigned.")

# Optional: save full transcript
if args.full_transcript_json:
    print(f"💾 Saving full transcript to {args.full_transcript_json} ...")
    with open(args.full_transcript_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print("✅ Full transcript saved.")

# ------------------- Chunking -------------------
def chunk_text_for_rag(transcript_segments, chunk_size=3):
    chunks = []
    for i in range(0, len(transcript_segments), chunk_size):
        segs = transcript_segments[i:i+chunk_size]
        text = " ".join([f"[{s['speaker']}] {s['text']}" for s in segs])
        start = segs[0]["start"]
        end = segs[-1]["end"]
        metadata = {
            "start": start,
            "end": end,
            "speakers": list({s["speaker"] for s in segs})
        }
        chunks.append((text, metadata))
    return chunks

print("📦 Chunking transcript for RAG ...")
chunks = chunk_text_for_rag(segments, chunk_size=args.chunk_size)
for _ in tqdm(chunks, desc="Chunks"):
    pass
print("✅ Chunking complete.")

# ------------------- Print or Return -------------------
print("=== Transcript Chunks for RAG ===")
for idx, (text, meta) in enumerate(chunks):
    print(f"--- Chunk {idx} ---")
    print(text)
    print(meta)
