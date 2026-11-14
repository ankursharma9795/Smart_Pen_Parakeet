
import time
import wave
import serial

PORT = "COM12"
BAUDRATE = 921600
SAMPLE_RATE = 16000
RECORD_SECONDS = 10
EXPECTED_BYTES = SAMPLE_RATE * RECORD_SECONDS * 2


def read_exactly(ser, count, timeout_seconds=20):
    data = bytearray()
    deadline = time.monotonic() + timeout_seconds

    while len(data) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Received {len(data)} of {count} audio bytes."
            )

        chunk = ser.read(min(4096, count - len(data)))
        if chunk:
            data.extend(chunk)

    return bytes(data)


def main():
    print(f"Opening {PORT} at {BAUDRATE} baud...")

    try:
        with serial.Serial(
            PORT, BAUDRATE, timeout=1, write_timeout=5
        ) as ser:

            # Allow the ESP32 to restart after opening the port.
            time.sleep(2)
            ser.reset_input_buffer()

            print("Requesting audio...")
            ser.write(b"R")
            ser.flush()

            # Wait for the AUD0 header.
            header = bytearray()
            deadline = time.monotonic() + 10

            while time.monotonic() < deadline:
                byte = ser.read(1)

                if byte:
                    header.extend(byte)

                    if len(header) > 4:
                        del header[:-4]

                    if header == b"AUD0":
                        break

            if header != b"AUD0":
                raise RuntimeError(
                    f"AUD0 header not received. Got: {bytes(header)!r}"
                )

            print("ESP32 synchronized!")
            print("Recording for 10 seconds... Speak near the microphone.")

            audio = read_exactly(ser, EXPECTED_BYTES)

            with wave.open("recording.wav", "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(audio)

            print("SUCCESS! Saved recording.wav")
            print("Open recording.wav to listen to your audio.")

    except serial.SerialException as error:
        print("Serial port error:", error)
        print("Close Arduino Serial Monitor and check COM12.")

    except Exception as error:
        print("ERROR:", error)


if __name__ == "__main__":
    main()
