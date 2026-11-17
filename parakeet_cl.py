import os
import re
import time
import queue
import shutil
import ctypes
import difflib
import threading
import subprocess
import webbrowser

from pathlib import Path
from collections import deque
from urllib.parse import quote_plus

import serial
import serial.tools.list_ports
import numpy as np
import torch

from transformers import AutoProcessor, AutoModelForCTC
from pynput.keyboard import Controller, Key, GlobalHotKeys


# ============================================================
# CONFIGURATION
# ============================================================

# ---- Hardware (unchanged) ----
COM_PORT = "COM6"
BAUD_RATE = 921600

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 320
CHUNK_BYTES = CHUNK_SAMPLES * 2

SERIAL_MARKER = b"AUD0"

# ---- Model ----
MODEL_NAME = "nvidia/parakeet-ctc-1.1b"
USE_FP16_ON_CUDA = True          # set False if you see odd output on GPU

# ---- Voice activity detection ----
VAD_THRESHOLD = 500              # minimum threshold (never lower than this)
MAX_VAD_THRESHOLD = 3000         # cap for the auto-calibrated threshold
NOISE_MARGIN = 2.0               # threshold = noise floor * this (if above minimum)
CALIBRATION_SECONDS = 1.5        # stay quiet for this long at startup

SILENCE_DURATION = 0.6
MIN_SPEECH_DURATION = 0.25       # counts only VOICED audio, not silence/pre-roll
MAX_PHRASE_DURATION = 12
PRE_ROLL_DURATION = 0.2
SPEECH_CONFIRMATION_CHUNKS = 5

# A "search ..." phrase waits this long for a continuation phrase,
# so "search inductance" ... pause ... "on wikipedia" still works.
SEARCH_MERGE_WINDOW = 1.5

# ---- Feature switches ----
REQUIRE_SLIDESHOW_FOR_SLIDE_KEYS = True
SLIDESHOW_TITLE_KEYWORDS = ("slide show", "slideshow")

# Automatic typing into AI sites that cannot be prefilled from the URL.
AI_WINDOW_TIMEOUT = 15.0     # seconds to wait for the browser to come to the front
AI_PAGE_LOAD_DELAY = 5.0     # extra seconds for the AI page to finish loading
ENABLE_DELETE_SPECIFIC_TEXT = False   # Ctrl+F method is unreliable outside Word

CHUNKS_PER_SECOND = SAMPLE_RATE / CHUNK_SAMPLES
SILENCE_CHUNKS = int(SILENCE_DURATION * CHUNKS_PER_SECOND)
MIN_VOICED_CHUNKS = int(MIN_SPEECH_DURATION * CHUNKS_PER_SECOND)
MAX_PHRASE_CHUNKS = int(MAX_PHRASE_DURATION * CHUNKS_PER_SECOND)
PRE_ROLL_CHUNKS = int(PRE_ROLL_DURATION * CHUNKS_PER_SECOND)
CALIBRATION_CHUNKS = int(CALIBRATION_SECONDS * CHUNKS_PER_SECOND)

keyboard = Controller()

audio_queue = queue.Queue(maxsize=500)
phrase_queue = queue.Queue()

running = threading.Event()
running.set()

system_active = False
capitalize_next = True
slideshow_paused = False
saved_window = None

SER = None


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading NVIDIA Parakeet model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dtype = (
    torch.float16
    if (device == "cuda" and USE_FP16_ON_CUDA)
    else torch.float32
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForCTC.from_pretrained(
    MODEL_NAME,
    dtype=model_dtype
)

model.to(device)
model.eval()

print(f"Model loaded on {device.upper()} ({str(model_dtype).split('.')[-1]})")


# ============================================================
# SERIAL CONNECTION
# ============================================================

def connect_esp32():

    ports = serial.tools.list_ports.comports()
    available_ports = [port.device for port in ports]

    if COM_PORT in available_ports:
        selected_port = COM_PORT
    elif available_ports:
        selected_port = available_ports[0]
        print(
            f"WARNING: {COM_PORT} not found. "
            f"Falling back to {selected_port}. "
            f"Available ports: {available_ports}"
        )
    else:
        raise RuntimeError("ESP32 not detected.")

    print(f"Connecting to ESP32 on {selected_port}...")

    ser = serial.Serial(
        selected_port,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(b"R")

    print("ESP32 connected. Audio streaming started.")

    return ser


def reconnect_esp32(old_serial, attempts=10, delay=3):

    try:
        old_serial.close()
    except Exception:
        pass

    for attempt in range(1, attempts + 1):

        if not running.is_set():
            return None

        print(f"Reconnecting to ESP32 ({attempt}/{attempts})...")

        try:
            return connect_esp32()
        except Exception as error:
            print("Reconnect failed:", error)
            time.sleep(delay)

    print("Could not reconnect to the ESP32. Stopping.")
    return None


# ============================================================
# AUDIO READER
# ============================================================

def read_exact(ser, size):

    data = bytearray()

    while len(data) < size and running.is_set():

        part = ser.read(size - len(data))

        if not part:
            continue

        data.extend(part)

    return bytes(data)


def audio_reader():

    global SER

    marker_buffer = bytearray()

    while running.is_set():

        try:

            byte = SER.read(1)

            if not byte:
                continue

            marker_buffer.extend(byte)

            if len(marker_buffer) > len(SERIAL_MARKER):
                del marker_buffer[0]

            if bytes(marker_buffer) == SERIAL_MARKER:

                audio_data = read_exact(SER, CHUNK_BYTES)

                if len(audio_data) != CHUNK_BYTES:
                    continue

                audio = np.frombuffer(
                    audio_data,
                    dtype=np.int16
                ).copy()

                try:
                    audio_queue.put_nowait(audio)
                except queue.Full:
                    pass

                marker_buffer.clear()

        except (serial.SerialException, OSError) as error:

            if not running.is_set():
                break

            print("Serial error:", error)

            SER = reconnect_esp32(SER)

            if SER is None:
                running.clear()
                break

            marker_buffer.clear()


# ============================================================
# VOICE ACTIVITY DETECTION
# ============================================================

def chunk_level(audio):

    # Peak level after removing DC offset.
    samples = audio.astype(np.float32)
    samples -= samples.mean()

    return float(np.max(np.abs(samples)))


def vad_worker():

    pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)

    speech_buffer = []

    speech_started = False
    speech_counter = 0
    silence_counter = 0
    voiced_chunks = 0

    # ---- Noise floor calibration ----
    print(
        f"Calibrating microphone noise "
        f"({CALIBRATION_SECONDS:.1f}s, stay quiet)..."
    )

    levels = []

    while running.is_set() and len(levels) < CALIBRATION_CHUNKS:

        try:
            audio = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        levels.append(chunk_level(audio))

    threshold = VAD_THRESHOLD

    if levels:

        noise_floor = float(np.percentile(levels, 90))

        threshold = min(
            max(VAD_THRESHOLD, noise_floor * NOISE_MARGIN),
            MAX_VAD_THRESHOLD
        )

        print(
            f"Noise floor: {noise_floor:.0f} | "
            f"VAD threshold: {threshold:.0f}"
        )

    # ---- Main detection loop ----
    while running.is_set():

        try:
            audio = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        is_speech = chunk_level(audio) > threshold

        if not speech_started:

            pre_roll.append(audio)

            if is_speech:
                speech_counter += 1
            else:
                speech_counter = 0

            if speech_counter >= SPEECH_CONFIRMATION_CHUNKS:

                speech_started = True
                speech_buffer = list(pre_roll)
                silence_counter = 0
                voiced_chunks = speech_counter

        else:

            speech_buffer.append(audio)

            if is_speech:
                silence_counter = 0
                voiced_chunks += 1
            else:
                silence_counter += 1

            if (
                silence_counter >= SILENCE_CHUNKS
                or len(speech_buffer) >= MAX_PHRASE_CHUNKS
            ):

                # Only voiced audio counts toward the minimum,
                # so short taps / pen clicks are ignored.
                if voiced_chunks >= MIN_VOICED_CHUNKS:

                    phrase = np.concatenate(speech_buffer)
                    phrase_queue.put(phrase)

                speech_started = False
                speech_counter = 0
                silence_counter = 0
                voiced_chunks = 0
                speech_buffer = []
                pre_roll.clear()


# ============================================================
# SPEECH RECOGNITION
# ============================================================

def transcribe_audio(audio):

    audio = audio.astype(np.float32) / 32768.0

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    # The processor key differs between model families
    # (input_features vs input_values), so pass everything through.
    prepared = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):

            value = value.to(device)

            if value.is_floating_point():
                value = value.to(model.dtype)

        prepared[key] = value

    with torch.inference_mode():

        outputs = model(**prepared)

        predicted_ids = torch.argmax(
            outputs.logits,
            dim=-1
        )

    transcription = processor.batch_decode(
        predicted_ids
    )[0]

    return transcription.strip()


def warm_up_model():

    print("Warming up model...")

    try:
        transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.int16))
    except Exception as error:
        print("Warm-up failed:", error)


# ============================================================
# KEYBOARD / WINDOW HELPERS
# ============================================================

def press_shortcut(*keys):

    for key in keys:
        keyboard.press(key)

    for key in reversed(keys):
        keyboard.release(key)


def tap(key):

    keyboard.press(key)
    keyboard.release(key)


def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip()
    )


def foreground_window_title():

    if os.name != "nt":
        return ""

    try:

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)

        user32.GetWindowTextW(hwnd, buffer, length + 1)

        return buffer.value

    except Exception:
        return ""


def get_foreground_hwnd():

    if os.name != "nt":
        return None

    try:
        return ctypes.windll.user32.GetForegroundWindow()
    except Exception:
        return None


def focus_window(hwnd):

    # Bring a specific window to the front. Returns True on success.
    if os.name != "nt" or not hwnd:
        return False

    try:

        user32 = ctypes.windll.user32

        if not user32.IsWindow(hwnd):
            return False

        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)      # SW_RESTORE

        # A brief Alt press lets Windows allow the focus change.
        user32.keybd_event(0x12, 0, 0, 0)
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(0x12, 0, 0x0002, 0)

        time.sleep(0.4)

        return user32.GetForegroundWindow() == hwnd

    except Exception as error:

        print("Focus error:", error)

        return False


def slideshow_active():

    title = foreground_window_title().lower()

    return any(
        keyword in title
        for keyword in SLIDESHOW_TITLE_KEYWORDS
    )


# ============================================================
# DICTATION FORMATTING
# ============================================================

SPOKEN_PUNCTUATION = [
    ("new paragraph", "\n\n"),
    ("new line", "\n"),
    ("next line", "\n"),
    ("full stop", "."),
    ("question mark", "?"),
    ("exclamation mark", "!"),
    ("exclamation point", "!"),
    ("semicolon", ";"),
    ("comma", ","),
]


def format_dictation(text):

    global capitalize_next

    for spoken, symbol in SPOKEN_PUNCTUATION:

        replacement = symbol if symbol.startswith("\n") else symbol + " "

        text = re.sub(
            r"\s*\b" + re.escape(spoken) + r"\b\s*",
            replacement,
            text
        )

    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = text.strip(" ")

    # Capitalize after sentence endings and new lines.
    text = re.sub(
        r"([.!?]\s+|\n)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text
    )

    # Capitalize the first letter if the previous text ended a sentence.
    if capitalize_next:

        text = re.sub(
            r"^([^\w]*)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            text
        )

    capitalize_next = bool(re.search(r"([.!?]|\n)\s*$", text))

    if text and not text.endswith(("\n", " ")):
        text += " "

    return text


def type_text(text):

    keyboard.type(text)


# ============================================================
# PRESENTATION CONTROL
# ============================================================

def next_slide():

    tap(Key.right)


def previous_slide():

    tap(Key.left)


def pause_slideshow():

    global slideshow_paused, saved_window

    print("Foreground window:", repr(foreground_window_title()))

    # "b" blackens the slide. Only send it during a real slideshow,
    # otherwise it would type a letter "b" into the document.
    if slideshow_active():

        # Remember the slideshow window so we can return to it later.
        saved_window = get_foreground_hwnd()

        if not slideshow_paused:

            tap("b")
            slideshow_paused = True

            time.sleep(0.3)


def resume_slideshow():

    global slideshow_paused

    print("Returning to slideshow...")

    returned = focus_window(saved_window)

    if not returned:

        # Fallback: Alt+Tab, held long enough for Windows to switch.
        keyboard.press(Key.alt_l)
        time.sleep(0.15)
        tap(Key.tab)
        time.sleep(0.15)
        keyboard.release(Key.alt_l)

        time.sleep(0.8)

    in_slideshow = (
        (saved_window is not None and get_foreground_hwnd() == saved_window)
        or slideshow_active()
    )

    if not in_slideshow:

        print(
            "Could not bring the slideshow to the front. "
            "Click it once, then say 'resume slideshow' again."
        )

        return

    if slideshow_paused:

        tap("b")
        slideshow_paused = False

    print("Slideshow resumed.")


# ============================================================
# WEBSITE SEARCH
# ============================================================

WEBSITE_URLS = {

    "wikipedia": "https://en.wikipedia.org/w/index.php?search=",

    "youtube": "https://www.youtube.com/results?search_query=",

    "google": "https://www.google.com/search?q=",

    "bing": "https://www.bing.com/search?q=",

    "duckduckgo": "https://duckduckgo.com/?q=",

    "google scholar": "https://scholar.google.com/scholar?q=",

    "amazon": "https://www.amazon.in/s?k=",

    "github": "https://github.com/search?q=",

    "reddit": "https://www.reddit.com/search/?q=",

    "stackoverflow": "https://stackoverflow.com/search?q=",

    "britannica": "https://www.britannica.com/search?query="
}


# "?q=" prefill works for these.
AI_WEBSITES = {

    "chatgpt": "https://chatgpt.com/",

    "copilot": "https://copilot.microsoft.com/",

    "claude": "https://claude.ai/new"
}

# Gemini cannot be prefilled from the URL, so the page is opened and
# the question is typed and sent automatically (see search_by_typing).
TYPED_AI_WEBSITES = {

    "gemini": "https://gemini.google.com/app"
}

WEBSITE_ALIASES = {

    "wiki": "wikipedia",
    "wikipedia.org": "wikipedia",
    "you tube": "youtube",
    "scholar": "google scholar",
    "google search": "google",
    "duck duck go": "duckduckgo",
    "git hub": "github",
    "stack overflow": "stackoverflow",
    "stackoverflow.com": "stackoverflow",
    "chat gpt": "chatgpt",
    "chat g p t": "chatgpt",
    "co pilot": "copilot",
    "microsoft copilot": "copilot",
    "google gemini": "gemini",
    "amazon india": "amazon"
}


def resolve_website(website):

    """Return a known site key, a domain, or None if not a site."""

    website = normalize_text(website)
    website = website.replace(" dot ", ".")
    website = WEBSITE_ALIASES.get(website, website)

    if (
        website in WEBSITE_URLS
        or website in AI_WEBSITES
        or website in TYPED_AI_WEBSITES
    ):
        return website

    domain = website.replace(" ", "")
    domain = domain.replace("https://", "").replace("http://", "")

    if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", domain):
        return domain

    return None


def copy_to_clipboard(text):

    try:

        subprocess.run(
            ["clip"],
            input=text.encode("utf-16"),
            check=True
        )

        return True

    except Exception as error:

        print("Clipboard error:", error)

        return False


def drain_phrase_queue():

    # Discard anything spoken while an automatic search was running,
    # so it is not typed into the wrong window afterwards.
    while True:

        try:
            phrase_queue.get_nowait()
        except queue.Empty:
            break


def search_by_typing(site, url, topic):

    webbrowser.open_new_tab(url)

    print(f"Opening {site}; the question will be typed automatically...")

    deadline = time.time() + AI_WINDOW_TIMEOUT

    while time.time() < deadline:

        if site in foreground_window_title().lower():
            break

        time.sleep(0.3)

    else:

        print(
            f"{site} did not come to the front within "
            f"{AI_WINDOW_TIMEOUT:.0f}s. Question NOT typed."
        )

        return False

    # Let the page finish loading so the prompt box is ready.
    time.sleep(AI_PAGE_LOAD_DELAY)

    # Safety: never type into some other window.
    if site not in foreground_window_title().lower():

        print(f"{site} lost focus. Question NOT typed.")

        return False

    keyboard.type(topic)

    time.sleep(0.5)

    tap(Key.enter)

    print(f"Sent to {site}: {topic}")

    return True


def open_website_search(topic, website):

    topic = topic.strip()

    encoded_topic = quote_plus(topic)

    if website in AI_WEBSITES:

        url = AI_WEBSITES[website] + "?q=" + encoded_topic

        webbrowser.open_new_tab(url)

        print(f"Opening {website} for: {topic}")

        return True

    if website in TYPED_AI_WEBSITES:

        search_by_typing(website, TYPED_AI_WEBSITES[website], topic)

        drain_phrase_queue()

        return True

    if website in WEBSITE_URLS:

        url = WEBSITE_URLS[website] + encoded_topic

        webbrowser.open_new_tab(url)

        print(f"Searching {topic} on {website}")

        return True

    # Generic domain search (website is a validated domain here).
    query = f"site:{website} {topic}"

    url = "https://www.google.com/search?q=" + quote_plus(query)

    webbrowser.open_new_tab(url)

    print(f"Searching {topic} on {website}")

    return True


# ============================================================
# STRICT SEARCH COMMANDS
# ============================================================

# The complete recognized sentence must BEGIN with "search".
#
#   "search hello"                       -> command
#   "i want to search hello"             -> normal typing
#   "search turn on the light"           -> Google search for the whole
#                                           phrase (not a site search)

def parse_site_search(text):

    """Return (topic, website) for 'search X on <known site>'."""

    match = re.fullmatch(
        r"search\s+(.+)\s+on\s+(.+)",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    topic = match.group(1).strip()
    website = resolve_website(match.group(2))

    if not topic or not website:
        return None

    return topic, website


def handle_universal_search(text):

    parsed = parse_site_search(text)

    if not parsed:
        return False

    topic, website = parsed

    pause_slideshow()
    time.sleep(0.5)

    return open_website_search(topic, website)


def handle_google_search(text):

    match = re.fullmatch(
        r"search\s+(.+)",
        text,
        re.IGNORECASE
    )

    if not match:
        return False

    topic = match.group(1).strip()

    if not topic:
        return False

    pause_slideshow()
    time.sleep(0.5)

    url = "https://www.google.com/search?q=" + quote_plus(topic)

    webbrowser.open_new_tab(url)

    print(f"Searching Google for: {topic}")

    return True


# ============================================================
# SMART FORMATTING (HEADINGS)
# ============================================================

def handle_smart_formatting(text):

    patterns = [
        r"(?:heading|title)\s+(?:is\s+)?(.+)",
        r"topic\s+is\s+(.+)"
    ]

    for pattern in patterns:

        match = re.fullmatch(pattern, text)

        if match:

            content = match.group(1).strip()

            if content:

                type_text(content.upper())

                return True

    return False


# ============================================================
# BULLET POINT GENERATION
# ============================================================

def handle_bullet_command(text):

    patterns = [
        r"(?:first|second|third|fourth|fifth|next) point\s+(?:is\s+)?(.+)",
        r"bullet(?: point)?\s+(.+)"
    ]

    for pattern in patterns:

        match = re.fullmatch(pattern, text)

        if match:

            content = match.group(1).strip()

            if content:

                content = content[0].upper() + content[1:]

                type_text("\u2022 " + content)

                return True

    return False


# ============================================================
# DELETE SYSTEM
# ============================================================

def delete_last_word():

    press_shortcut(Key.ctrl_l, Key.backspace)


def delete_specific_text(text):

    # Fragile outside Word: Ctrl+F does not select text everywhere.
    press_shortcut(Key.ctrl_l, "f")

    time.sleep(0.3)

    type_text(text)

    time.sleep(0.3)

    tap(Key.esc)
    tap(Key.backspace)


# ============================================================
# UNDO / REDO
# ============================================================

def undo_action():

    press_shortcut(Key.ctrl_l, "z")


def redo_action():

    press_shortcut(Key.ctrl_l, "y")


# ============================================================
# APPLICATION LAUNCHER
# ============================================================

# Direct protocol links: work even if the app is missing from the index.
APP_PROTOCOLS = {

    "whatsapp": "whatsapp:",
    "spotify": "spotify:",
    "telegram": "tg:",
    "discord": "discord:",
    "slack": "slack:"
}


START_MENU_INDEX = {}   # normalized name -> ("file", Path) or ("appid", str)

# Common ways the speech model mishears app names.
APP_SPOKEN_ALIASES = {

    "whats app": "whatsapp",
    "what's app": "whatsapp",
    "what app": "whatsapp",
    "watsapp": "whatsapp",
    "whatsapp web": "whatsapp",
    "ms word": "word",
    "ms excel": "excel",
    "ms powerpoint": "powerpoint",
    "power point": "powerpoint",
    "v s code": "visual studio code",
    "file explorer": "explorer",
    "command prompt": "cmd",
    "task manager": "task manager"
}


def build_start_menu_index():

    START_MENU_INDEX.clear()

    # 1. Classic Start Menu shortcuts
    folders = [

        Path(os.environ.get("APPDATA", ""))
        / "Microsoft/Windows/Start Menu/Programs",

        Path(os.environ.get("PROGRAMDATA", ""))
        / "Microsoft/Windows/Start Menu/Programs"

    ]

    for folder in folders:

        if not folder.exists():
            continue

        for extension in ("*.lnk", "*.url", "*.exe"):

            try:

                for candidate in folder.rglob(extension):

                    key = normalize_text(candidate.stem)

                    if key.startswith("uninstall"):
                        continue

                    START_MENU_INDEX.setdefault(
                        key,
                        ("file", candidate)
                    )

            except Exception:
                pass

    # 2. Every app Windows lists in Start, including Microsoft Store
    #    (UWP) apps such as WhatsApp, which have no .lnk shortcut.
    try:

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-StartApps | ForEach-Object "
                "{ $_.Name + '|' + $_.AppID }"
            ],
            capture_output=True,
            text=True,
            timeout=40,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        for line in result.stdout.splitlines():

            if "|" not in line:
                continue

            name, app_id = line.rsplit("|", 1)

            key = normalize_text(name)
            app_id = app_id.strip()

            if key and app_id and not key.startswith("uninstall"):

                # Prefer the AppID entry (works for Store and classic apps).
                START_MENU_INDEX[key] = ("appid", app_id)

    except Exception as error:

        print("Get-StartApps failed:", error)

    print(f"Indexed {len(START_MENU_INDEX)} applications.")


def find_start_menu_application(app_name):

    name = normalize_text(app_name)

    if len(name) < 3:
        return None

    if name in START_MENU_INDEX:
        return START_MENU_INDEX[name]

    words = name.split()

    for stem, entry in START_MENU_INDEX.items():

        stem_words = stem.split()

        if stem.startswith(name) or all(w in stem_words for w in words):
            return entry

    # Fuzzy match for small recognition errors.
    close = difflib.get_close_matches(
        name,
        list(START_MENU_INDEX.keys()),
        n=1,
        cutoff=0.8
    )

    if close:
        return START_MENU_INDEX[close[0]]

    return None


def launch_index_entry(entry):

    kind, value = entry

    try:

        if kind == "appid":

            subprocess.Popen(
                ["explorer.exe", "shell:AppsFolder\\" + value]
            )

        else:

            os.startfile(str(value))

        return True

    except Exception as error:

        print("Application launch error:", error)

        return False


def launch_any_application(app_name):

    app_name = normalize_text(app_name)
    app_name = APP_SPOKEN_ALIASES.get(app_name, app_name)

    entry = find_start_menu_application(app_name)

    # Newly installed app? Refresh the index once and retry.
    if entry is None:

        build_start_menu_index()
        entry = find_start_menu_application(app_name)

    if entry:
        print(f"Matched {app_name!r} -> {entry[0]}: {entry[1]}")

        if launch_index_entry(entry):
            return True
    else:
        print(f"{app_name!r} not found in application index.")

    protocol = APP_PROTOCOLS.get(app_name)

    if protocol:

        try:
            os.startfile(protocol)
            print(f"Opened {app_name} via {protocol}")
            return True
        except Exception as error:
            print("Protocol launch error:", error)

    executable = shutil.which(app_name)

    if executable:

        try:
            subprocess.Popen([executable])
            return True
        except Exception as error:
            print("Application launch error:", error)

    suggestions = difflib.get_close_matches(
        app_name,
        list(START_MENU_INDEX.keys()),
        n=3,
        cutoff=0.5
    )

    print(f"No application found for: {app_name}")

    if suggestions:
        print("Did you mean:", ", ".join(suggestions))

    return False


def open_application(text):

    match = re.fullmatch(
        r"open\s+(.+)",
        text
    )

    if not match:
        return False

    app_name = match.group(1).strip()

    # "open whatsapp app" -> "whatsapp"
    stripped = re.sub(r"\s+(application|app)$", "", app_name)

    if stripped:
        app_name = stripped

    app_name = APP_SPOKEN_ALIASES.get(app_name, app_name)

    websites = {

        "youtube": "https://www.youtube.com",

        "gmail": "https://mail.google.com",

        "wikipedia": "https://www.wikipedia.org",

        "google": "https://www.google.com",

        "github": "https://github.com"

    }

    if app_name in websites:

        webbrowser.open_new_tab(websites[app_name])
        return True

    known_commands = {

        "chrome": "chrome",
        "brave": "brave",
        "vscode": "code",
        "vs code": "code",
        "visual studio code": "code",
        "settings": "ms-settings:",
        "calculator": "calc",
        "notepad": "notepad",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
        "arduino": "arduino",
        "matlab": "matlab",
        "zoom": "zoom"

    }

    if app_name in known_commands:

        command = known_commands[app_name]

        try:

            if command == "ms-settings:":
                os.startfile(command)

            else:

                subprocess.Popen(
                    ["cmd", "/c", "start", "", command],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )

            return True

        except Exception:
            pass

    return launch_any_application(app_name)


# ============================================================
# COMMAND HANDLER
# ============================================================

def handle_new_functions(text):

    global system_active

    text = normalize_text(text)

    if not text:
        return True

    # 0. Sleep / ignore repeated wake word
    if text in (
        "go to sleep",
        "sleep mode",
        "sleep",
        "stop listening"
    ):

        system_active = False
        print("System Sleeping")

        return True

    if text == "wake up":
        return True

    # 1. Search topic on a particular website
    if handle_universal_search(text):
        return True

    # 2. Search topic on Google
    if handle_google_search(text):
        return True

    # 3. Slide navigation (only inside a running slideshow)
    if text in ("next", "next slide", "previous", "previous slide"):

        if slideshow_active() or not REQUIRE_SLIDESHOW_FOR_SLIDE_KEYS:

            if text.startswith("next"):
                next_slide()
            else:
                previous_slide()

            return True

    # 4. Heading
    if handle_smart_formatting(text):
        return True

    # 5. Bullet points
    if handle_bullet_command(text):
        return True

    # 6. Delete last word
    if text in (
        "delete word",
        "remove word",
        "erase word"
    ):

        delete_last_word()
        return True

    # 7. Delete specific text
    match = re.fullmatch(
        r"(?:delete word|remove word|delete text|remove text|erase text)\s+(.+)",
        text
    )

    if match:

        if ENABLE_DELETE_SPECIFIC_TEXT:
            delete_specific_text(match.group(1))
        else:
            print(
                "Delete-specific-text is disabled "
                "(set ENABLE_DELETE_SPECIFIC_TEXT = True to enable)."
            )

        return True

    # 8. Undo
    if text in (
        "undo",
        "undo last",
        "undo that"
    ):

        undo_action()
        return True

    # 9. Redo
    if text in (
        "redo",
        "redo last"
    ):

        redo_action()
        return True

    # 10. Resume slideshow ("slideshow" is often heard as "slide show")
    if (
        re.fullmatch(
            r"(?:please\s+)?"
            r"(?:resume|continue|restart|return to|go back to|back to)\s+"
            r"(?:the\s+)?"
            r"(?:slide\s*show|presentation|slides?|power\s*point|ppt)",
            text
        )
        or (slideshow_paused and text in ("resume", "continue", "go back"))
    ):

        resume_slideshow()
        return True

    # 11. Open applications / websites.
    # Only consumed if something was actually launched;
    # otherwise the sentence is typed as normal dictation.
    if text.startswith("open "):

        if open_application(text):
            return True

    return False


# ============================================================
# RECOGNIZED SPEECH HANDLER
# ============================================================

def handle_recognized_text(text):

    global system_active

    text = normalize_text(text)

    if not text:
        return

    print("Recognized:", text)

    # Sleeping mode
    if not system_active:

        if "wake up" in text:

            system_active = True

            print("System Activated")

        return

    # Active mode
    if handle_new_functions(text):
        return

    # No command matched: type the sentence with punctuation/capitals.
    type_text(format_dictation(text))


# ============================================================
# TRANSCRIPTION WORKER
# ============================================================

def starts_like_search(text):

    text = normalize_text(text)

    return text == "search" or text.startswith("search ")


def transcription_worker():

    pending_text = ""
    pending_time = 0.0

    while running.is_set():

        try:
            audio = phrase_queue.get(timeout=0.2)

        except queue.Empty:

            # Merge window expired: run the held search phrase.
            if pending_text and time.time() - pending_time > SEARCH_MERGE_WINDOW:

                held = pending_text
                pending_text = ""

                try:
                    handle_recognized_text(held)
                except Exception as error:
                    print("Command error:", error)

            continue

        try:

            text = transcribe_audio(audio)

            if not text:
                continue

            merged = False

            if pending_text:

                text = pending_text + " " + text
                pending_text = ""
                merged = True

            # Hold an incomplete search command briefly so a pause in
            # the middle of it doesn't split it into two phrases.
            if (
                system_active
                and not merged
                and starts_like_search(text)
                and not parse_site_search(normalize_text(text))
            ):

                pending_text = text
                pending_time = time.time()

                continue

            handle_recognized_text(text)

        except Exception as error:

            print("Transcription error:", error)


# ============================================================
# STOP HOTKEY
# ============================================================

def stop_program():

    global system_active

    print("\nStopping Smart Pen...")

    system_active = False
    running.clear()


# ============================================================
# MAIN
# ============================================================

def main():

    global SER

    print("\n===================================")
    print("          SMART PEN SYSTEM")
    print("===================================")

    print("State: SLEEPING")
    print('Say "wake up" to activate, "go to sleep" to pause.')
    print("Press Ctrl + Alt + Q to stop.")

    print("\nSearch commands:")
    print("Search hello")
    print("Search inductance on Wikipedia")
    print("Search inductance on Gemini")
    print("Search inductance on ChatGPT")
    print("Search inductance on Copilot")
    print("Search inductance on Claude")

    print("\nTyping commands:")
    print("Heading <text>  |  First point <text>  |  Bullet <text>")
    print("Comma / Full stop / Question mark / New line / New paragraph")
    print("Delete word  |  Undo  |  Redo")
    print("===================================\n")

    build_start_menu_index()

    warm_up_model()

    SER = connect_esp32()

    threads = [

        threading.Thread(
            target=audio_reader,
            daemon=True
        ),

        threading.Thread(
            target=vad_worker,
            daemon=True
        ),

        threading.Thread(
            target=transcription_worker,
            daemon=True
        )

    ]

    for thread in threads:
        thread.start()

    try:

        with GlobalHotKeys({
            "<ctrl>+<alt>+q": stop_program
        }):

            while running.is_set():
                time.sleep(0.2)

    except KeyboardInterrupt:

        stop_program()

    finally:

        running.clear()

        try:
            if SER is not None:
                SER.close()
        except Exception:
            pass

        print("Smart Pen stopped.")


if __name__ == "__main__":
    main()