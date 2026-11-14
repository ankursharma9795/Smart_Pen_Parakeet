import os
import sys

# Ensure CUDA DLLs are found for GPU acceleration on Windows
cuda_dll_dirs = [
    r"C:\Users\rajvi\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
]
for d in cuda_dll_dirs:
    if os.path.exists(d):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

from flask import Flask, request
from faster_whisper import WhisperModel

app = Flask(__name__)

model_size = "tiny"
print(f"Loading faster-whisper model: {model_size}")
try:
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Model successfully loaded on NVIDIA GPU (CUDA).")
except Exception as e:
    print(f"CUDA initialization failed ({e}), falling back to CPU...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("Model loaded on CPU.")

@app.route('/audio', methods=['POST'])
def audio():
    audio_data = request.get_data()
    
    if not audio_data:
        return "No audio data received", 400

    filename = "esp32_audio.wav"
    with open(filename, "wb") as f:
        f.write(audio_data)

    print(f"Received audio ({len(audio_data)} bytes), transcribing on GPU...")
    
    try:
        segments, info = model.transcribe(
            filename,
            beam_size=5,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        transcription = " ".join(s.text.strip() for s in segments).strip()
        print(f"Transcription: {transcription}")
        
        return (transcription if transcription else "No speech detected") + "\n"
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"Error: {e}\n", 500

if __name__ == '__main__':
    print("Starting Whisper Flask Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
