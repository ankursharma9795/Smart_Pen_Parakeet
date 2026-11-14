import os
import sys

# 1. Register PyTorch CUDA DLLs first before any other modules
torch_lib = r"C:\Users\arunu\OneDrive\Desktop\nvidia testing\whisper_env\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import torch
import time
import wave
import serial
import serial.tools.list_ports
import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC

COM_PORT = "COM12"
BAUD_RATE = 921600
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
RECORD_SECONDS = 30

def find_esp32_port():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    if COM_PORT in ports:
        return COM_PORT
    if ports:
        return ports[0]
    return None

def record_from_esp32(port, duration_sec=30, output_wav="live_recording.wav"):
    expected_bytes = duration_sec * SAMPLE_RATE * BYTES_PER_SAMPLE
    print(f"\n[MIC] Opening {port} at {BAUD_RATE} baud...")

    with serial.Serial(port, BAUD_RATE, timeout=1, write_timeout=5) as ser:
        # Give ESP32 2 seconds to reboot upon opening serial port
        print("[MIC] Initializing ESP32 connection (2s)...")
        time.sleep(2)
        ser.reset_input_buffer()

        print("=" * 65)
        print(f" >>> SPEAK NOW! Recording {duration_sec}s from INMP441 mic... <<<")
        print("=" * 65)

        # Send start command 'R' to ESP32
        ser.write(b"R")
        ser.flush()

        # Wait for AUD0 header
        header = bytearray()
        deadline = time.monotonic() + 5
        header_found = False

        while time.monotonic() < deadline:
            byte = ser.read(1)
            if byte:
                header.extend(byte)
                if len(header) > 4:
                    del header[:-4]
                if header == b"AUD0":
                    header_found = True
                    break

        if not header_found:
            raise RuntimeError("Did not receive AUD0 synchronization header from ESP32.")

        # Read exact audio bytes
        raw_audio = bytearray()
        t_deadline = time.monotonic() + duration_sec + 5

        while len(raw_audio) < expected_bytes:
            if time.monotonic() >= t_deadline:
                print("\n[MIC] Warning: Timeout waiting for all audio bytes.")
                break
            chunk = ser.read(min(4096, expected_bytes - len(raw_audio)))
            if chunk:
                raw_audio.extend(chunk)
                pct = int((len(raw_audio) / expected_bytes) * 100)
                print(f"\r[MIC] Recording: {pct}% ({len(raw_audio)}/{expected_bytes} bytes)", end="", flush=True)

        # Stop recording on ESP32
        ser.write(b"S")
        ser.flush()

    print(f"\n[MIC] Captured {len(raw_audio)} bytes ({len(raw_audio)/(SAMPLE_RATE*BYTES_PER_SAMPLE):.2f}s).")

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)

    return output_wav

def main():
    print("=" * 65)
    print(" NVIDIA Parakeet-CTC 1.1B - Live ESP32 Speech-To-Text")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"-> Compute device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("-> Compute device: CPU (Warning: CUDA not detected, running on CPU)")

    print("\nLoading Parakeet-CTC 1.1B model into VRAM...")
    t0 = time.time()
    model_id = "nvidia/parakeet-ctc-1.1b"
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCTC.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()

    if device == "cuda":
        print(f"-> Model ready in {time.time() - t0:.2f}s | VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    else:
        print(f"-> Model ready in {time.time() - t0:.2f}s")

    port = find_esp32_port()
    if not port:
        print("Error: No serial port detected. Make sure ESP32 is plugged in.")
        return

    print(f"-> Target ESP32 port: {port}")
    print("\nReady! Press Enter to start a recording, or type 'q' then Enter to quit.")

    while True:
        try:
            choice = input("\n[Enter to Record / 'q' to Quit]: ").strip().lower()
            if choice == "q":
                print("Exiting...")
                break

            wav_path = record_from_esp32(port, duration_sec=RECORD_SECONDS)

            print("\n[ASR] Transcribing with NVIDIA Parakeet-CTC 1.1B on GPU...")
            speech, sr = sf.read(wav_path)
            if speech.ndim > 1:
                speech = speech.mean(axis=1)

            t_infer = time.time()
            inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device, dtype=dtype if v.is_floating_point() else None) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0].strip()
            elapsed = time.time() - t_infer

            print("\n" + "=" * 65)
            print(" PARAKEET TRANSCRIPTION:")
            print(f" >>> \"{transcription}\" <<<")
            print(f" (Latency: {elapsed:.2f}s | Duration: {len(speech)/sr:.1f}s)")
            print("=" * 65)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
