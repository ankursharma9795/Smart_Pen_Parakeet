# Smart_Pen_Parakeet
NVIDIA Parakeet-CTC 1.1B -- Live ESP32 Speech-to-Text
A Python-based speech-to-text application that captures audio from an
INMP441 microphone connected to an ESP32, transfers the audio to a
Windows computer over USB serial, and transcribes it using NVIDIA
Parakeet-CTC 1.1B with PyTorch.

The application uses CUDA when a compatible NVIDIA GPU and CUDA-enabled
PyTorch installation are available. Otherwise, it falls back to CPU
execution.
Features
- Captures mono audio from an INMP441 microphone through the ESP32.
- Transfers raw audio data to the computer over serial communication.
- Uses a synchronization header (AUD0) before receiving audio data.
- Records a fixed-duration audio segment (30 seconds by default).
- Saves the received audio as a WAV file.
- Transcribes the recording using nvidia/parakeet-ctc-1.1b.
- Automatically selects CUDA or CPU based on PyTorch availability.
- Keeps the speech-recognition model loaded while allowing repeated
  recordings.
Hardware Requirements
- ESP32 development board
- INMP441 omnidirectional MEMS digital microphone
- USB data cable
- Windows computer
- NVIDIA GPU for CUDA acceleration (optional)
The ESP32 firmware must be configured to receive the serial commands
and transmit audio in the format expected by this Python application.

Software Requirements
- Windows
- Python 3.10 (the configured PyTorch DLL path in the script uses a
  Python 3.10 virtual environment)
- A compatible PyTorch installation
- NVIDIA GPU drivers and a compatible CUDA-enabled PyTorch build for
  GPU acceleration
- Internet connection on the first run to download the model and
  processor from Hugging Face
Python packages used:
- torch
- transformers
- soundfile
- pyserial
- numpy (commonly required by the audio/ML package stack)
Project Setup
1. Create and activate a virtual environment
Open PowerShell in the project directory:
py -3.10 -m venv whisper_env
.\whisper_env\Scripts\Activate.ps1
If PowerShell blocks script execution for the current session, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\whisper_env\Scripts\Activate.ps1
2. Install dependencies
Install a PyTorch build that matches your system and CUDA environment.
Follow the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Then install the remaining packages:
pip install transformers soundfile pyserial numpy
Check whether PyTorch can detect CUDA:
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
3. Configure the PyTorch DLL path
At the beginning of the Python script, update torch_lib so it points
to the torch\lib directory inside the active virtual environment.

Example:
torch_lib = r"C:\path\to\project\whisper_env\Lib\site-packages\torch\lib"
The script adds this directory to PATH and registers it with
os.add_dll_directory() before importing PyTorch. This helps Windows
locate the PyTorch DLL files.
4. Connect the ESP32
1. Connect the ESP32 to the computer using a USB data cable.
2. Check the serial port assigned by Windows Device Manager.
3. Update COM_PORT in the script if necessary.
4. Make sure the ESP32 firmware uses the same baud rate and serial
   protocol described below.
Configuration
The main settings are defined near the top of the script:
  Setting                                    Default Description
  COM_PORT                                 COM12 Preferred ESP32
serial port

  BAUD_RATE                               921600 Serial communication
baud rate

  SAMPLE_RATE                              16000 Audio sample rate in
Hz

  BYTES_PER_SAMPLE                             2 16-bit audio = 2
bytes per sample

  RECORD_SECONDS                              30 Recording duration in
seconds

  model_id              nvidia/parakeet-ctc-1.1b Hugging Face model
identifier
output_wav                  live_recording.wav Output WAV filename
The application first checks whether COM12 is available. If it is not,
it uses the first serial port detected by pyserial.
ESP32 Serial Communication Protocol
The Python application expects the ESP32 firmware to follow this
sequence:
1. The Python application opens the serial port at 921600 baud.
2. It waits approximately two seconds for the ESP32 to initialize, then
   clears the serial input buffer.
3. Python sends the single-byte command R to start audio
   transmission.
4. The ESP32 sends the four-byte synchronization header:
   AUD0
5. After the header, the ESP32 transmits raw audio bytes.
6. Python reads up to the expected number of bytes:
   duration × sample rate × bytes per sample
   With the default settings:
   30 × 16000 × 2 = 960000 bytes
7. Python sends the single-byte command S to stop recording.
8. The received audio is saved as a mono, 16-bit WAV file at 16 kHz.
Important: The audio payload must match the WAV settings used by the
Python script: mono, signed 16-bit PCM, and 16 kHz. The ESP32 firmware
must not insert text messages or other data into the binary audio
payload after the AUD0 header.
Running the Application
Save the Python code as, for example:
live_esp32_parakeet.py
With the virtual environment activated and the ESP32 connected, run:
python live_esp32_parakeet.py
On startup, the application:
1. Checks whether CUDA is available.
2. Loads the NVIDIA Parakeet-CTC 1.1B processor and model.
3. Detects the ESP32 serial port.
4. Waits for the user to start a recording.
Use the prompt as follows:
- Press Enter to record and transcribe an audio segment.
- Type q and press Enter to quit.
The first run may take longer because the model files need to be
downloaded. The model remains loaded while the application is running,
so subsequent recordings do not reload it.
Output
Each recording is saved as:
live_recording.wav
After recording, the application passes the audio to the Parakeet model
and prints the recognized text in the terminal, along with the inference
latency and audio duration.

The WAV file is overwritten by the next recording because the same
output filename is used each time.
Troubleshooting
CUDA is not detected
- Confirm that an NVIDIA GPU is installed.
- Install a PyTorch build compatible with your GPU and CUDA
  environment.
- Run the CUDA availability check shown in the setup section.
- If CUDA remains unavailable, the script will run on CPU.
No serial port is detected
- Check the USB cable; some cables support charging only.
- Check the port in Windows Device Manager.
- Confirm that the ESP32 USB-to-serial driver is installed if
  required.
- Update COM_PORT or reconnect the board and run the script again.
AUD0 synchronization header is not received
- Confirm that the ESP32 firmware recognizes the R command.
- Confirm that the ESP32 sends exactly AUD0 before the binary audio
  payload.
- Check that the baud rate is 921600 on both sides.
- Make sure no other application is using the ESP32 serial port.
Recording times out or the audio is incomplete
- Check the USB connection and serial communication stability.
- Confirm that the ESP32 transmits the expected audio format and data
  rate.
- Verify that the firmware sends enough audio bytes for the configured
  duration.
- Check whether the serial buffer or firmware transmission logic is
  limiting throughput.
Audio is distorted or transcription is inaccurate
- Verify that the microphone is wired and powered correctly.
- Confirm that the firmware sends mono, 16-bit PCM audio at 16 kHz.
- Check the microphone input level and background noise.
- Ensure that the sample rate and sample format in the ESP32 firmware
  match the Python configuration.
Windows reports a PyTorch DLL loading error
- Verify that torch_lib points to the active environment's
  torch\lib folder.
- Make sure the virtual environment is activated.
- Reinstall the appropriate PyTorch build if required.
Notes and Limitations
- The current script records fixed-duration audio segments; it does
  not perform continuous streaming transcription.
- The model is loaded once at startup and reused for each recording.
- The script uses torch.argmax() over the model logits and decodes
  the resulting token IDs.
- The recording loop catches common runtime exceptions and allows the
  user to try another recording.
- The serial protocol and audio format must be implemented
  consistently in the ESP32 firmware.
Model
NVIDIA Parakeet-CTC 1.1B

Hugging Face model: https://huggingface.co/nvidia/parakeet-ctc-1.1b
License
Add the license applicable to your project and verify the licenses of
the model, dependencies, and any ESP32 firmware components before
distributing the project.
