import os
import sys
import time

# Ensure PyTorch CUDA DLLs are found on Windows
torch_lib = r"C:\Users\rajvi\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC

def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "recording.wav"

    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)

    print("=" * 65)
    print(" NVIDIA Parakeet-CTC 1.1B ASR Engine")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Using compute device: {device.upper()}")
    if device == "cuda":
        print(f"-> GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"-> VRAM: {total_vram:.1f} GB")

    model_id = "nvidia/parakeet-ctc-1.1b"
    print(f"\n[1/3] Loading Processor & Model weights ({model_id})...")
    print("      (First run will download ~2.2 GB model files from Hugging Face)")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)

    # Use float16 on GPU to keep VRAM ~2.2 GB (well within RTX 2050 4GB limit)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCTC.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    print(f"-> Model loaded in {time.time() - t0:.2f} seconds.")

    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"-> VRAM allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    print(f"\n[2/3] Loading audio: {audio_path}")
    speech, sample_rate = sf.read(audio_path)
    
    # If stereo, convert to mono
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    duration = len(speech) / sample_rate
    print(f"-> Duration: {duration:.2f}s | Sample rate: {sample_rate} Hz")

    print(f"\n[3/3] Running Parakeet-CTC 1.1B inference...")
    t_start = time.time()

    inputs = processor(speech, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device, dtype=dtype if v.is_floating_point() else None) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    infer_time = time.time() - t_start

    print("\n" + "=" * 65)
    print(" TRANSCRIPTION RESULT")
    print("=" * 65)
    print(f"\"{transcription.strip()}\"")
    print("-" * 65)
    print(f"Inference time: {infer_time:.2f}s (Real-Time Factor: {infer_time / max(duration, 0.01):.3f}x)")
    print("=" * 65)

if __name__ == "__main__":
    main()

