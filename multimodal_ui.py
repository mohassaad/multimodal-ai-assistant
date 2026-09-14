"""
============================================================
 Mini AI Assistant — Week 4 Final Task
 Agent with Tool Selection (LangGraph) + Gradio UI
 ---- LIQUID GLASS UI VARIANT ----

 FIXED: X-ray classification speed
   1) image was resized to (160,160) but the model was trained on
      (128,128) -> TensorFlow silently retraced its computation graph
      on every single upload (very slow). Now resized to (128,128) to
      match training.
   2) model.predict() spins up a full tf.data pipeline meant for
      batches/datasets. For a single image, calling the model directly
      (model(arr, training=False)) skips all of that overhead.
   3) Added a one-time "warm-up" inference right after the model loads,
      so the one-time graph-compilation cost happens at startup instead
      of on the user's first real upload.

 FIXED: X-ray predictions on real uploads
   4) Uploads were squashed into a 128x128 square, and screenshots kept
      their black bars — nothing like the training images. prepare_xray()
      now trims dark borders and cuts out the centre square, using the
      exact same decoding and code as train_xray.py (retrain to match).
============================================================
"""

import os
import re
import shutil
import tempfile
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from groq import Groq
import tensorflow as tf
from PIL import Image
from pypdf import PdfReader
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import gradio as gr


# ============================================================
# SETUP
# ============================================================
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("❌ GROQ_API_KEY not found.")

groq_client = Groq(api_key=GROQ_API_KEY)

AGENT_MODEL = "openai/gpt-oss-20b"
TEXT_MODEL  = "openai/gpt-oss-20b"
STT_MODEL   = "whisper-large-v3"

# --- FIX: this must match the size the model was TRAINED on ---
# train_xray.py builds MobileNetV2 with input_shape=(128, 128, 3),
# so inference has to use the same (128, 128) size, not (160, 160).
IMG_SIZE = (128, 128)

# dark-border trimming — must match train_xray.py
BORDER_LEVEL    = 20.0   # pixels darker than this (0-255) count as border
BORDER_FRACTION = 0.05   # a row/column with fewer than 5% brighter pixels is border

# train_xray.py copies each newly trained model into this folder.
# The model is loaded ONCE here, so restart the app after retraining.
CAT_DOG_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat_dog_model.keras")
cat_dog_model = tf.keras.models.load_model(
    CAT_DOG_MODEL_PATH,
    safe_mode=False,
    custom_objects={"preprocess_input": preprocess_input},
)
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]

# Friendly display names for the UI (keeps the raw folder names above intact)
CLASS_DISPLAY = {
    "COVID19": "COVID-19",
    "NORMAL": "Normal",
    "PNEUMONIA": "Pneumonia",
    "TURBERCULOSIS": "Tuberculosis",
}

# --- FIX: warm up the model once at startup ---
# The very first call to a freshly-loaded Keras model always pays a
# one-time graph-compilation cost. Doing it here (at import time) means
# that cost happens once, when the app starts, instead of on the user's
# first real X-ray upload.
print(" Warming up X-ray model...")
_ = cat_dog_model(tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3)), training=False)
print(" X-ray model READY.")


# ============================================================
# CLEANER
# ============================================================
def clean(text: str) -> str:
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()


# ============================================================
# TOOL 1 — TEXT
# ============================================================
@tool
def text_tool(text: str) -> str:
    """Analyze a plain text sentence or paragraph. Use this when the user
    provides raw TEXT (not a file). It will classify the text into a category
    (Sports, Politics, Technology, Health, Entertainment) AND extract key
    information (Person, Organization, Location, Date, Event).
    Input: the sentence or paragraph to analyze."""

    classify_resp = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a text classifier. Classify the text into EXACTLY ONE of:\n"
                    "Sports, Politics, Technology, Health, Entertainment.\n"
                    "Reply with ONLY the category name, nothing else."
                ),
            },
            {"role": "user", "content": f"Text: {text}"},
        ],
        temperature=0,
    )
    category = clean(classify_resp.choices[0].message.content)

    extract_resp = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an information extraction assistant.\n"
                    "Extract the following fields from the text:\n"
                    "- Person\n- Organization\n- Location\n- Date\n- Event\n"
                    "Reply in this EXACT format:\n"
                    "Person: ...\nOrganization: ...\nLocation: ...\nDate: ...\nEvent: ...\n"
                    "If a field is not found, write N/A."
                ),
            },
            {"role": "user", "content": f"Text: {text}"},
        ],
        temperature=0,
    )
    extracted = clean(extract_resp.choices[0].message.content)

    return (
        f" Category: {category}\n\n"
        f"🔍 Extracted Information:\n{extracted}"
    )


# ============================================================
# TOOL 2 — DOCUMENT (PDF)
# ============================================================
@tool
def document_tool(file_path: str) -> str:
    """Extract and summarize information from a DOCUMENT file, especially a
    PDF such as an invoice, report, or form. Use this when the user uploads
    or points to a .pdf file. Input: the path to the PDF file."""

    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"

    reader = PdfReader(file_path)
    raw_text = "\n".join((page.extract_text() or "") for page in reader.pages)

    if not raw_text.strip():
        return "❌ This PDF has no extractable text (it may be a scanned image)."

    resp = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document analysis assistant. Summarize the key information "
                    "from the document. If it is an invoice, extract: vendor, invoice number, "
                    "date, line items, and total amount. Be clear and structured. "
                    "Use plain text only — no HTML tags, no asterisks, no markdown symbols."
                ),
            },
            {"role": "user", "content": raw_text[:6000]},
        ],
        temperature=0,
    )
    return clean(resp.choices[0].message.content)


# ============================================================
# TOOL 3 — IMAGE
# ============================================================

# --- FIX: prepare uploads EXACTLY like train_xray.py prepares training images.
# This is an identical copy of prepare_xray() in train_xray.py — change both
# or neither. The old code squashed every upload into a 128x128 square, so a
# wide screenshot with black bars looked nothing like the training images.
def prepare_xray(rgb):
    """RGB image (h, w, 3) -> grayscale -> dark borders trimmed -> centre square
    (no stretching) -> IMG_SIZE, as 3 identical channels in [0, 255]."""
    # grayscale kills colour/tint leakage; MobileNetV2 still needs 3 channels
    gray = tf.image.rgb_to_grayscale(tf.cast(rgb, tf.float32))

    # trim near-black bars (screenshot letterboxing, scanner margins)
    content = tf.cast(gray[..., 0] > BORDER_LEVEL, tf.float32)
    rows = tf.where(tf.reduce_mean(content, axis=1) > BORDER_FRACTION)[:, 0]
    cols = tf.where(tf.reduce_mean(content, axis=0) > BORDER_FRACTION)[:, 0]
    big_enough = tf.logical_and(tf.size(rows) * 3 >= tf.shape(gray)[0],   # only crop if at least
                                tf.size(cols) * 3 >= tf.shape(gray)[1])   # 1/3 of the picture is left
    trimmed = tf.cond(big_enough,
                      lambda: gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1],
                      lambda: gray)

    # cut out the centre square instead of stretching, then shrink.
    # (not padding: black bars would appear on the wide child X-rays only,
    #  and the model would learn "bars = Normal/Pneumonia" — a new shortcut)
    side = tf.minimum(tf.shape(trimmed)[0], tf.shape(trimmed)[1])
    square = tf.image.resize_with_crop_or_pad(trimmed, side, side)
    small = tf.image.resize(square, IMG_SIZE, antialias=True)
    return tf.image.grayscale_to_rgb(small)


@tool
def image_tool(file_path: str) -> str:
    """Classify a CHEST X-RAY image into one of four conditions
    (Normal, Pneumonia, COVID-19, or Tuberculosis) using a trained
    deep-learning model. Use this when the user uploads an image file
    (.jpg / .jpeg / .png) of a chest X-ray. Input: the path to the image."""

    if not os.path.exists(file_path):
        return f"❌ Image not found: {file_path}"

    # Decode with TensorFlow, exactly like train_xray.py, so the model gets the
    # same pixels it was trained on (PIL decodes JPEGs slightly differently).
    # PIL is only the fallback for formats TensorFlow can't read.
    try:
        rgb = tf.io.decode_image(tf.io.read_file(file_path), channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError:
        rgb = tf.keras.utils.img_to_array(Image.open(file_path).convert("RGB"))

    # prepare_xray() output is always the TRAINING size (128,128), so TensorFlow
    # never sees a new input shape (no slow graph retracing).
    arr = tf.expand_dims(prepare_xray(rgb), 0)  # batch dimension: (1, 128, 128, 3)

    # --- FIX: call the model directly instead of model.predict().
    # predict() spins up a tf.data pipeline meant for batches/datasets,
    # which is unnecessary overhead for a single image. A direct call
    # (model(arr, training=False)) is the fast path for one-off inference.
    preds = cat_dog_model(arr, training=False).numpy()[0]

    idx = int(tf.argmax(preds))
    label = CLASS_DISPLAY.get(CLASS_NAMES[idx], CLASS_NAMES[idx])
    confidence = float(preds[idx])

    # show the full probability breakdown across all four conditions
    ranking = sorted(
        ((CLASS_DISPLAY.get(CLASS_NAMES[i], CLASS_NAMES[i]), float(preds[i]))
         for i in range(len(CLASS_NAMES))),
        key=lambda x: x[1],
        reverse=True,
    )
    breakdown = "\n".join(f"   • {name}: {p:.1%}" for name, p in ranking)

    return (
        f"🫁 Predicted: {label} ({confidence:.1%} confidence)\n\n"
        f"Full breakdown:\n{breakdown}\n\n"
        f"Always consult a qualified radiologist."
    )


# ============================================================
# TOOL 4 — AUDIO
# ============================================================
@tool
def audio_tool(file_path: str) -> str:
    """Transcribe an AUDIO file — convert spoken words into written text
    (Speech-to-Text). Use this when the user uploads an audio file
    (.mp3 / .wav / .m4a) and wants to know what was said.
    Input: the path to the audio file."""

    if not os.path.exists(file_path):
        return f"❌ Audio not found: {file_path}"

    with open(file_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            file=(os.path.basename(file_path), f.read()),
            model=STT_MODEL,
        )
    return f"🎙️ Transcription:\n{result.text}"


# ============================================================
# AGENT
# ============================================================
SYSTEM_PROMPT = (
    "You are a Mini AI Assistant with four specialized tools:\n"
    "- text_tool: for analyzing plain text (classification + info extraction)\n"
    "- document_tool: for reading and summarizing PDF files\n"
    "- image_tool: for classifying a chest X-ray as Normal, Pneumonia, COVID-19, or Tuberculosis\n"
    "- audio_tool: for transcribing audio files to text\n\n"
    "When the user sends a request:\n"
    "1. Identify what type of input it is (text, PDF path, image path, audio path)\n"
    "2. Call exactly ONE tool that matches\n"
    "3. Return the tool's result as your final answer, clearly formatted."
)

tools_list = [text_tool, document_tool, image_tool, audio_tool]

agent = create_react_agent(
    ChatGroq(model=AGENT_MODEL, temperature=0),
    tools=tools_list,
    prompt=SYSTEM_PROMPT,
)

def run_agent(message: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return clean(result["messages"][-1].content)


# ============================================================
# HANDLERS
# ============================================================
def handle_text(text: str):
    if not text.strip():
        return "⚠️ Please enter some text first."
    return run_agent(f"Analyze this text: {text}")

def handle_pdf(pdf_file):
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."
    return run_agent(f"Extract information from this PDF document: {pdf_file}")

def handle_image(image_file):
    if image_file is None:
        return "⚠️ Please upload an image first."
    return image_tool.invoke({"file_path": image_file})

def handle_audio(audio_file):
    if audio_file is None:
        return "⚠️ Please upload an audio file first."
    return run_agent(f"Transcribe this audio file: {audio_file}")


# ============================================================
# CSS
# ============================================================
LIQUID_GLASS_CSS = r"""

@import url('https://fonts.googleapis.com/css2?family=Chango&family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --glass-bg: rgba(255, 255, 255, 0.10);
    --glass-border: rgba(255, 255, 255, 0.35);
    --glass-shadow: 0 8px 32px rgba(15, 15, 40, 0.35);
    --accent-a: #7dd3fc;
    --accent-b: #c084fc;
    --accent-c: #f472b6;
    --text-main: #f5f6fa;
    --text-dim: rgba(245, 246, 250, 0.65);

    /* button / tab system (dark defaults) */
    --surface-2: #0b0f2a;
    --btn-border: rgba(255, 255, 255, 0.35);
    --btn-hover: rgba(255, 255, 255, 0.10);

    --page-bg:
        radial-gradient(circle at 15% 20%, rgba(125, 211, 252, 0.35), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(192, 132, 252, 0.35), transparent 45%),
        radial-gradient(circle at 30% 85%, rgba(244, 114, 182, 0.30), transparent 45%),
        radial-gradient(circle at 80% 80%, rgba(94, 234, 212, 0.25), transparent 45%),
        linear-gradient(135deg, #0b0f2a 0%, #131a3d 45%, #1a0f33 100%);
}

html[data-theme="light"] {
    color-scheme: light;
    --glass-bg: rgba(255, 255, 255, 0.62);
    --glass-border: rgba(45, 45, 80, 0.16);
    --glass-shadow: 0 8px 32px rgba(70, 75, 110, 0.16);
    --text-main: #182033;
    --text-dim: rgba(24, 32, 51, 0.68);

    --surface-2: #eef3ff;
    --btn-border: rgba(24, 32, 51, 0.20);
    --btn-hover: rgba(24, 32, 51, 0.06);

    --page-bg:
        radial-gradient(circle at 15% 20%, rgba(125, 211, 252, 0.38), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(192, 132, 252, 0.32), transparent 45%),
        radial-gradient(circle at 30% 85%, rgba(244, 114, 182, 0.23), transparent 45%),
        radial-gradient(circle at 80% 80%, rgba(94, 234, 212, 0.20), transparent 45%),
        linear-gradient(135deg, #eef4ff 0%, #f8f3ff 48%, #fff4fa 100%);
}

gradio-app,
.gradio-container,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.app {
    background: var(--page-bg) !important;
    background-attachment: fixed !important;
    background-size: 200% 200%, 200% 200%, 200% 200%, 200% 200%, 100% 100% !important;
}
.gradio-container {
    font-family: 'Inter', 'Poppins', sans-serif !important;
    min-height: 100vh !important;
    background: var(--page-bg) !important;
    background-attachment: fixed !important;
    background-size: 200% 200%, 200% 200%, 200% 200%, 200% 200%, 100% 100% !important;
    animation: liquidDrift 22s ease-in-out infinite alternate;
    color: var(--text-main) !important;
}
.gradio-container .block:not(.glass-card),
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gap,
.gradio-container .html-container,
.gradio-container .prose {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container,
.gradio-container * {
    cursor: none !important;
}
.lga-cursor {
    position: fixed; top: 0; left: 0;
    width: 26px; height: 26px;
    pointer-events: none;
    z-index: 2147483647;
    opacity: 1;
    will-change: transform;
}
.lga-cursor svg {
    width: 100%; height: 100%; display: block;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.55));
}

@keyframes liquidDrift {
    0%   { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 0% 0%; }
    50%  { background-position: 30% 20%, 70% 30%, 20% 80%, 80% 70%, 0% 0%; }
    100% { background-position: 60% 40%, 40% 60%, 40% 40%, 60% 60%, 0% 0%; }
}

.gradio-container .header-box,
.header-box {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 16px 24px 30px;
    margin-bottom: 22px;
    border-radius: 28px;
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--glass-shadow), inset 0 1px 0 rgba(255,255,255,0.25) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    position: relative;
    overflow: hidden;
}
.header-box::before {
    content: "";
    position: absolute;
    top: -60%; left: -20%;
    width: 140%; height: 220%;
    background: linear-gradient(120deg, transparent 40%, rgba(255,255,255,0.18) 50%, transparent 60%);
    transform: rotate(8deg);
    animation: sheen 8s linear infinite;
    pointer-events: none;
}
@keyframes sheen {
    0% { transform: translateX(-30%) rotate(8deg); }
    100% { transform: translateX(30%) rotate(8deg); }
}
.header-box h1 {
    font-family: 'Cooper Black', 'Cooper Std Black', 'CooperBlack', 'Chango', Georgia, serif !important;
    font-weight: 400 !important;
    font-size: 3rem !important;
    line-height: 1.0 !important;
    margin: 0 0 6px 0 !important;
    background: none !important;
    -webkit-text-fill-color: #ffffff !important;
    color: #ffffff !important;
    letter-spacing: 0.5px;
}
html[data-theme="light"] .header-box h1 {
    -webkit-text-fill-color: #182033 !important;
    color: #182033 !important;
}
.header-box .header-sub {
    transform: translateY(-3px);
}
.header-box .agent-subtitle,
.header-box .agent-description {
    position: relative;
    color: var(--text-dim);
}
.header-box .agent-subtitle {
    margin: 4px 0 0 0 !important;
}
.header-box .agent-description {
    margin: 2px 0 0 0 !important;
}

.theme-toggle-slot {
    position: fixed !important;
    top: 16px !important;
    right: 18px !important;
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto !important;
}

.theme-toggle {
    --step: 0.5s;
    --ease: linear(
        0 0%, 0.2342 12.49%, 0.4374 24.99%, 0.6093 37.49%, 0.6835 43.74%,
        0.7499 49.99%, 0.8086 56.25%, 0.8593 62.5%, 0.9023 68.75%,
        0.9375 75%, 0.9648 81.25%, 0.9844 87.5%, 0.9961 93.75%, 1 100%
    );
    --trail-ease: linear(
        0 0%, 0.0039 6.25%, 0.0156 12.5%, 0.0352 18.75%, 0.0625 25%,
        0.0977 31.25%, 0.1407 37.5%, 0.1914 43.74%, 0.2499 49.99%,
        0.3164 56.25%, 0.3906 62.5%, 0.5625 75%, 0.7656 87.5%, 1 100%
    );
    --offset: calc(var(--step) * 0.5);
    --glow: hsl(182 90% 92%);
    --button-dark: hsl(220 27% 6%);
    --button-light: hsl(0 0% 97%);

    font-size: 13.5px !important;
    width: auto !important;
    height: 3em !important;
    aspect-ratio: 1.8 / 1 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 3em !important;
    background: transparent !important;
    position: relative !important;
    cursor: pointer !important;
    box-shadow: none !important;
    outline: none !important;
    overflow: visible !important;
    -webkit-tap-highlight-color: transparent;
}

.theme-toggle::before {
    content: '';
    position: absolute; inset: -0.35em;
    border-radius: 3em;
    z-index: -1;
    pointer-events: none;
    box-shadow: 0 0 0.9em 0.15em rgba(255,255,255,0.45);
    opacity: 0.9;
    transition: opacity var(--step) var(--ease);
}
html[data-theme="light"] .theme-toggle::before {
    box-shadow: 0 0 0.9em 0.15em rgba(210,215,235,0.55);
}

.theme-toggle :is(.socket, .face) { position: absolute; border-radius: 3em; }

.theme-toggle .socket {
    inset: 0;
    background: hsl(0 0% 0%);
    box-shadow: 0 0 12px 3px rgba(255,255,255,0.20);
    transition: background-color var(--step) var(--ease), box-shadow var(--step) var(--ease);
}
html[data-theme="light"] .theme-toggle .socket,
.theme-toggle[aria-pressed="true"] .socket {
    background: hsl(0 0% 97%);
    box-shadow: 0 0 10px 2px rgba(190,195,215,0.22);
}
.theme-toggle .socket-shadow {
    position: absolute; inset: 0; opacity: 0; border-radius: inherit;
    box-shadow: 0 0.075em 0.1em 0 white;
    transition: opacity var(--step) var(--ease);
}

.theme-toggle .face { inset: 0.15em; transition: scale var(--step) var(--ease); }

.theme-toggle .face-shadow,
.theme-toggle .face-shadow::after,
.theme-toggle .face-shadow::before { position: absolute; inset: 0; border-radius: inherit; }
.theme-toggle .face-shadow::after,
.theme-toggle .face-shadow::before { content: ''; }
.theme-toggle .face-shadow::before { background: black; }
.theme-toggle .face-shadow::after { background: white; scale: 0.5; }
.theme-toggle .face-shadow::after,
.theme-toggle .face-shadow::before {
    transition: opacity var(--step) var(--ease), translate var(--step) var(--ease),
                filter var(--step) var(--ease), scale var(--step) var(--ease);
}

.theme-toggle .face-plate {
    position: absolute; inset: 0; border-radius: inherit;
    box-shadow: 0 0 0.15em 0 hsl(0 0% 100% / 0.10) inset;
    background: conic-gradient(from 45deg, #0000, hsl(0 0% 100% / 0.05)), var(--button-dark);
    transition: background var(--step) var(--ease);
}
html[data-theme="light"] .theme-toggle .face-plate,
.theme-toggle[aria-pressed="true"] .face-plate {
    background: conic-gradient(from 45deg, #0000, hsl(0 0% 100% / 0.05)), var(--button-light);
}

.theme-toggle .face-glowdrop {
    position: absolute; inset: 0; border-radius: inherit; scale: 0;
    transition: scale var(--step) var(--ease);
}
.theme-toggle .face-glowdrop::after,
.theme-toggle .face-glowdrop::before {
    content: ''; height: 50%; aspect-ratio: 1; background: #fff;
    filter: blur(0.1em); position: absolute; z-index: -1; border-radius: 50%;
}
.theme-toggle .face-glowdrop::before { left: 4%; width: 56%; translate: 0 -25%; }
.theme-toggle .face-glowdrop::after { bottom: 0; right: 12%; width: 34%; translate: 0 20%; }

.theme-toggle .face-shine {
    position: absolute; inset: 0; opacity: 0; border-radius: 3em;
    transition: opacity var(--step) var(--ease);
}
.theme-toggle .face-shine-shadow {
    position: absolute; inset: 0; border-radius: inherit;
    mask: conic-gradient(from 0deg, #fff 90deg, #0000 110deg 200deg, #fff 215deg 280deg, #0000 315deg);
    box-shadow: 0.075em 0 0.025em -0.025em hsl(0 0% 0% / 0.5) inset,
                -0.075em -0.05em 0.025em -0.025em hsl(0 0% 0% / 0.5) inset;
}
.theme-toggle .face-shine::before {
    content: ''; position: absolute; inset: 0.05em; border-radius: 3em;
    box-shadow: 0 -0.05em 0.025em -0.025em hsl(0 0% 50% / 0.5) inset,
                -0.025em 0.05em 0.025em -0.025em hsl(0 0% 100% / 0.5) inset;
}
.theme-toggle .face-shine::after {
    content: ''; position: absolute; inset: 0; border-radius: 3em;
    background: conic-gradient(from 45deg, #0000, hsl(0 0% 100% / 0.25));
}

.theme-toggle .face-glows {
    position: absolute; inset: -0.075em; opacity: 0; border-radius: inherit;
    mix-blend-mode: plus-lighter; filter: blur(0.125em); z-index: 20;
    mask: conic-gradient(from 280deg, #0000, #fff 20deg 45deg, #0000 95deg),
          conic-gradient(from 110deg, #0000, #fff 20deg, #0000 95deg);
    transition: opacity var(--step) var(--ease);
}
.theme-toggle .face-glows div {
    position: absolute; inset: 0; border-radius: inherit;
    filter: blur(0.0625em); border: 0.1em solid white;
}

.theme-toggle .face svg {
    width: 25%; position: absolute; top: 50%; left: 50%;
    translate: -52% -48%; overflow: visible !important;
}
.theme-toggle .face svg path { transform-box: fill-box; transform-origin: center center; }

.theme-toggle .glow-path { fill: var(--glow); stroke: var(--glow); opacity: 1; stroke-width: 0; }

.theme-toggle .trail-holder { z-index: 2; filter: blur(0.156em); }
.theme-toggle .trail-holder .trail { stroke-width: 4; }

.theme-toggle .inner-face { fill: hsl(230 5% 80%); }
.theme-toggle .outline { stroke: #000; transition: stroke var(--step) var(--ease); }
.theme-toggle .inner-bg { fill: black; transition: fill var(--step) var(--ease); }

.theme-toggle .trail {
    stroke: #2CC6FE; stroke-linecap: round;
    stroke-dasharray: 10 80; stroke-dashoffset: 10; opacity: 0;
    transition-property: stroke-dashoffset, opacity;
    transition-duration: calc(var(--step) * 3), calc(var(--step) * 0.5);
    transition-delay: var(--offset), calc(var(--offset) + (var(--step) * 2.5));
    transition-timing-function: var(--ease), var(--trail-ease);
}

.theme-toggle .glow {
    z-index: 3;
    filter: drop-shadow(0 0 0.2em var(--glow));
    will-change: opacity;
    opacity: 1;
    transition-property: opacity;
    transition-duration: 1.25s;
    transition-delay: var(--offset);
    transition-timing-function: var(--trail-ease);
}

.theme-toggle[aria-pressed="true"] .face { scale: 1.12; }
.theme-toggle[aria-pressed="true"] .outline { stroke: hsl(0 0% 30%); }
.theme-toggle[aria-pressed="true"] .inner-bg { fill: hsl(0 0% 20%); }
.theme-toggle[aria-pressed="true"] .socket { box-shadow: 0 0 10px 2px rgba(190,195,215,0.22); }
.theme-toggle[aria-pressed="true"] .face-glowdrop { scale: 1; }
.theme-toggle[aria-pressed="true"]::before { opacity: 1; box-shadow: 0 0 1em 0.18em rgba(255,255,255,0.5); }
.theme-toggle[aria-pressed="true"] .face-shadow::before {
    translate: -15% 55%; filter: blur(1em); opacity: 0.35;
}
.theme-toggle[aria-pressed="true"] .face-shadow::after { filter: blur(0.5em); scale: 1; }
.theme-toggle[aria-pressed="true"] .socket-shadow,
.theme-toggle[aria-pressed="true"] .face-shine { opacity: 1; }
.theme-toggle[aria-pressed="true"] .face-glows { opacity: 0; }
.theme-toggle[aria-pressed="true"] .trail {
    transition: stroke-dashoffset 0s; opacity: 1; stroke-dashoffset: -70;
}
.theme-toggle[aria-pressed="true"] .glow {
    opacity: 0; transition-property: opacity;
    transition-duration: var(--step); transition-delay: 0s;
    transition-timing-function: var(--ease);
}

.theme-toggle:active .socket { box-shadow: 0 0 8px 2px rgba(255,255,255,0.12); }
.theme-toggle:active .face { scale: 0.99; }

.gradio-container .glass-card,
.glass-card {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    padding: 22px !important;
    box-shadow: var(--glass-shadow), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    backdrop-filter: blur(20px) saturate(150%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
    transition: box-shadow 0.4s ease, transform 0.4s ease;
}
.glass-card:hover {
    box-shadow: 0 12px 40px rgba(15, 15, 40, 0.45), inset 0 1px 0 rgba(255,255,255,0.25);
}
.tool-label {
    display: inline-block; font-family: 'Poppins', sans-serif; font-size: .72rem;
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
    padding: 5px 14px; margin-bottom: 10px; border-radius: 999px;
    color: var(--text-main);
    background: linear-gradient(90deg, rgba(125,211,252,.35), rgba(192,132,252,.35));
    border: 1px solid rgba(255,255,255,.3);
}

.tabs > .tab-nav,
div[role="tablist"] {
    display: flex !important;
    justify-content: center !important;
    align-items: flex-end !important;
    gap: 18px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 6px 0 10px !important;
    margin-top: 14px !important;
    margin-bottom: 14px !important;
    overflow: visible !important;
}
.tabs > .tab-nav button,
div[role="tablist"] button {
    min-width: 0 !important;
    height: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-main) !important;
    background: transparent !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 16px !important;
    padding: 10px 18px !important;
    margin: 0 !important;
    box-shadow: none !important;
    position: relative;
    overflow: visible !important;
    transform-origin: center bottom;
    transition: background .2s ease, color .2s ease, box-shadow .2s ease,
                transform .14s cubic-bezier(.2,.8,.2,1) !important;
    will-change: transform;
}
.tabs > .tab-nav button .tab-inner,
div[role="tablist"] button .tab-inner {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    line-height: 1 !important;
}
.tab-icon {
    width: 18px !important; height: 18px !important; flex: 0 0 auto !important;
    display: block !important; color: currentColor !important;
}
.tab-icon .f { fill: currentColor; }
.tab-icon .s {
    fill: none; stroke: currentColor; stroke-width: 1.5;
    stroke-linecap: round; stroke-linejoin: round;
}
.tabs > .tab-nav button:hover,
div[role="tablist"] button:hover {
    background: var(--btn-hover) !important;
    color: var(--text-main) !important;
}
.tabs > .tab-nav button.selected,
div[role="tablist"] button.selected,
div[role="tablist"] button[aria-selected="true"] {
    background: var(--text-main) !important;
    color: var(--surface-2) !important;
    border-color: transparent !important;
    border-bottom: 2px solid var(--surface-2) !important;
    box-shadow: 0 0 0 0.5px var(--btn-border), 0 4px 14px rgba(0,0,0,.18) !important;
    font-weight: 600 !important;
}
.tabs > .tab-nav button::after,
div[role="tablist"] button::after {
    display: none !important;
}
.tabs > .tab-nav button:active,
div[role="tablist"] button:active {
    transform: scale(.94);
    transition-duration: .06s !important;
}

html:not([data-theme="light"]) div[role="tablist"] button:not(.selected):not([aria-selected="true"]),
html:not([data-theme="light"]) .tabs > .tab-nav button:not(.selected):not([aria-selected="true"]) {
    background: rgba(255,255,255,0.09) !important;
    border: 1px solid rgba(255,255,255,0.55) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.28) !important;
}
html:not([data-theme="light"]) div[role="tablist"] button:not(.selected):not([aria-selected="true"]):hover,
html:not([data-theme="light"]) .tabs > .tab-nav button:not(.selected):not([aria-selected="true"]):hover {
    background: rgba(255,255,255,0.16) !important;
}

button.primary,
.glass-btn button {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important; letter-spacing: .3px;
    color: var(--text-main) !important;
    background: transparent !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
    transition: background .2s ease, color .2s ease, transform .1s ease, box-shadow .2s ease !important;
}
button.primary:hover,
.glass-btn button:hover {
    background: var(--btn-hover) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,.2) !important;
}
button.primary:active,
.glass-btn button:active {
    transform: scale(.96);
    box-shadow: none !important;
    transition-duration: .06s !important;
}

html[data-theme="light"] div[role="tablist"] button,
html[data-theme="light"] .tabs > .tab-nav button,
html[data-theme="light"] button.primary,
html[data-theme="light"] .glass-btn button {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(24,32,51,0.30) !important;
    color: #0e1626 !important;
    box-shadow: 0 2px 8px rgba(70,75,110,0.12) !important;
}
html[data-theme="light"] div[role="tablist"] button:hover,
html[data-theme="light"] .tabs > .tab-nav button:hover,
html[data-theme="light"] button.primary:hover,
html[data-theme="light"] .glass-btn button:hover {
    background: rgba(255,255,255,0.92) !important;
    color: #0e1626 !important;
}
html[data-theme="light"] div[role="tablist"] button.selected,
html[data-theme="light"] div[role="tablist"] button[aria-selected="true"] {
    background: var(--text-main) !important;
    color: #ffffff !important;
    border-color: transparent !important;
}

.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container .output-box textarea {
    background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: 16px !important; color: var(--text-main) !important;
}
html[data-theme="light"] .gradio-container textarea,
html[data-theme="light"] .gradio-container input[type="text"],
html[data-theme="light"] .gradio-container .output-box textarea {
    background: rgba(255,255,255,.58) !important; border-color: rgba(24,32,51,.12) !important;
}
.gradio-container label span { color: var(--text-dim) !important; font-weight: 500 !important; }
div[data-testid="file"], .upload-box, .image-container, .audio-container {
    background: rgba(255,255,255,.05) !important; border: 1.5px dashed rgba(255,255,255,.3) !important;
    border-radius: 18px !important; backdrop-filter: blur(10px);
}
html[data-theme="light"] div[data-testid="file"],
html[data-theme="light"] .upload-box,
html[data-theme="light"] .image-container,
html[data-theme="light"] .audio-container {
    background: rgba(255,255,255,.48) !important; border-color: rgba(24,32,51,.14) !important;
}
div[data-testid="file"]:hover, .upload-box:hover {
    border-color: var(--accent-a) !important; background: rgba(255,255,255,.09) !important;
}
.glass-card p, .glass-card li, .glass-card span { color: var(--text-dim) !important; }
.footer-box {
    text-align: center; margin-top: 24px; padding: 14px; border-radius: 999px;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.15);
    backdrop-filter: blur(14px); color: var(--text-dim); font-size: .8rem;
}
html[data-theme="light"] .footer-box {
    background: rgba(255,255,255,.48); border-color: rgba(24,32,51,.12);
}

#lga-loader-overlay {
    position: fixed; inset: 0; z-index: 100001;
    display: none;
    align-items: center; justify-content: center;
    background: rgba(6, 9, 24, 0.55);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}
html[data-theme="light"] #lga-loader-overlay {
    background: rgba(230, 235, 250, 0.55);
}
#lga-loader-overlay.show { display: flex; }

.loader-wrapper {
    position: relative;
    display: flex; align-items: center; justify-content: center;
    height: 120px; width: auto; margin: 2rem;
    font-family: 'Poppins', sans-serif;
    font-size: 1.6em; font-weight: 600;
    user-select: none; color: #fff;
    scale: 2;
}
html[data-theme="light"] .loader-wrapper { color: #10192b; }
.loader {
    position: absolute; top: 0; left: 0;
    height: 100%; width: 100%; z-index: 1;
    background-color: transparent;
    mask: repeating-linear-gradient(90deg, transparent 0, transparent 6px, black 7px, black 8px);
}
.loader::after {
    content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        radial-gradient(circle at 50% 50%, #ff0 0%, transparent 50%),
        radial-gradient(circle at 45% 45%, #f00 0%, transparent 45%),
        radial-gradient(circle at 55% 55%, #0ff 0%, transparent 45%),
        radial-gradient(circle at 45% 55%, #0f0 0%, transparent 45%),
        radial-gradient(circle at 55% 45%, #00f 0%, transparent 45%);
    mask: radial-gradient(circle at 50% 50%, transparent 0%, transparent 10%, black 25%);
    animation: transform-animation 2s infinite alternate, opacity-animation 4s infinite;
    animation-timing-function: cubic-bezier(0.6, 0.8, 0.5, 1);
}
@keyframes transform-animation { 0% { transform: translate(-55%);} 100% { transform: translate(55%);} }
@keyframes opacity-animation { 0%,100% { opacity: 0;} 15% { opacity: 1;} 65% { opacity: 0;} }
.loader-letter { display: inline-block; opacity: 0; animation: loader-letter-anim 4s infinite linear; z-index: 2; }
.loader-letter:nth-child(1) { animation-delay: 0.1s; }
.loader-letter:nth-child(2) { animation-delay: 0.205s; }
.loader-letter:nth-child(3) { animation-delay: 0.31s; }
.loader-letter:nth-child(4) { animation-delay: 0.415s; }
.loader-letter:nth-child(5) { animation-delay: 0.521s; }
.loader-letter:nth-child(6) { animation-delay: 0.626s; }
.loader-letter:nth-child(7) { animation-delay: 0.731s; }
.loader-letter:nth-child(8) { animation-delay: 0.837s; }
.loader-letter:nth-child(9) { animation-delay: 0.942s; }
.loader-letter:nth-child(10) { animation-delay: 1.047s; }
@keyframes loader-letter-anim {
    0% { opacity: 0; }
    5% { opacity: 1; text-shadow: 0 0 4px #fff; transform: scale(1.1) translateY(-2px); }
    20% { opacity: 0.2; }
    100% { opacity: 0; }
}

.gradio-container .progress-text,
.gradio-container .eta-bar,
.gradio-container .wrap.default,
.gradio-container .meta-text-center,
.gradio-container .meta-text { display: none !important; }
.gradio-container .generating { border: none !important; animation: none !important; }

@media (max-width: 900px) {
    .tabs > .tab-nav, div[role="tablist"] { flex-wrap: wrap !important; }
    .theme-toggle-slot { top: 8px !important; right: 8px !important; }
}

.gradio-container { position: relative; }
.gradio-container > * { position: relative; z-index: 1; }
#lga-orb-field {
    position: absolute; inset: 0; z-index: 0;
    pointer-events: none; overflow: hidden;
}
#lga-orb-field .orb {
    position: absolute; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #3a4454, #0b0f17);
    box-shadow: inset -10px -10px 20px rgba(0,0,0,0.8), 0 10px 20px rgba(0,0,0,0.5);
    animation: floatBubble 8s ease-in-out infinite;
    will-change: transform;
}
html[data-theme="light"] #lga-orb-field .orb {
    background: radial-gradient(circle at 30% 30%, #ffffff, #c3cede);
    box-shadow: inset -8px -8px 18px rgba(150,160,190,0.45), 0 10px 24px rgba(120,130,160,0.22);
}
@keyframes floatBubble {
    0%, 100% { transform: translateY(0) translateX(0) scale(1); }
    33%      { transform: translateY(-25px) translateX(12px) scale(1.03); }
    66%      { transform: translateY(15px) translateX(-10px) scale(0.97); }
}
#lga-orb-field .orb-1 { width:200px; height:200px; top:8%;    left:12%; }
#lga-orb-field .orb-2 { width:90px;  height:90px;  top:6%;    left:52%; filter:blur(7px); opacity:.45; animation-delay:1s; }
#lga-orb-field .orb-3 { width:70px;  height:70px;  top:22%;   right:14%; filter:blur(5px); opacity:.5; animation-delay:2s; }
#lga-orb-field .orb-4 { width:320px; height:320px; bottom:10%; right:4%; filter:blur(5px); opacity:.6; animation-delay:1.5s; }
#lga-orb-field .orb-5 { width:220px; height:220px; bottom:8%; left:8%;  filter:blur(5px); opacity:.5; animation-delay:2s; }
#lga-orb-field .orb-6 { width:90px;  height:90px;  top:52%;   left:5%;  filter:blur(6px); opacity:.45; animation-delay:2.5s; }
#lga-orb-field .orb-7 { width:120px; height:120px; top:46%;   left:48%; filter:blur(5px); opacity:.5; animation-delay:2s; }

"""


# ============================================================
# THEME TOGGLE HTML
# ============================================================
THEME_TOGGLE_HTML = """
<div class="theme-toggle-slot">
  <button aria-pressed="false" class="theme-toggle" type="button" id="lga-theme-toggle">
    <div class="socket">
      <div class="socket-shadow"></div>
    </div>
    <div class="face">
      <div class="face-shadow"></div>
      <div class="face-glowdrop"></div>
      <div class="face-plate"></div>
      <div class="face-shine">
        <div class="face-shine-shadow"></div>
      </div>
      <div class="face-glows"><div></div></div>
      <svg class="glow" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path class="glow-path" stroke-width="0"
          d="M9.8815 1.36438L9.88141 1.36429C9.70639 1.18942 9.48342 1.07041 9.24073 1.02235C8.99803 0.974286 8.74653 0.999323 8.51808 1.09429L8.51753 1.09452C4.54484 2.75146 1.75 6.6732 1.75 11.25C1.75 17.3262 6.67489 22.25 12.75 22.25C14.9217 22.2501 17.0448 21.6075 18.852 20.4032C20.6591 19.1989 22.0695 17.4868 22.9055 15.4825L22.9058 15.4818C23.0007 15.2532 23.0256 15.0015 22.9774 14.7587C22.9291 14.5159 22.8099 14.2929 22.6348 14.118C22.4597 13.9431 22.2366 13.8241 21.9937 13.7761C21.7509 13.7281 21.4993 13.7533 21.2708 13.8484L21.2707 13.8485C20.2346 14.2801 19.1231 14.5016 18.0007 14.5H18C15.7457 14.5 13.5837 13.6045 11.9896 12.0104C10.3955 10.4163 9.5 8.25433 9.5 5.99999L9.5 5.99927C9.49838 4.8769 9.71983 3.76541 10.1515 2.72938C10.2468 2.50072 10.2721 2.24888 10.224 2.00584C10.1759 1.76281 10.0567 1.53954 9.8815 1.36438Z"/>
      </svg>
      <svg class="trail-holder" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path class="trail" stroke="#2CC6FE" stroke-linecap="round" stroke-dasharray="7 80" stroke-dashoffset="40"
          d="M9.8815 1.36438L9.88141 1.36429C9.70639 1.18942 9.48342 1.07041 9.24073 1.02235C8.99803 0.974286 8.74653 0.999323 8.51808 1.09429L8.51753 1.09452C4.54484 2.75146 1.75 6.6732 1.75 11.25C1.75 17.3262 6.67489 22.25 12.75 22.25C14.9217 22.2501 17.0448 21.6075 18.852 20.4032C20.6591 19.1989 22.0695 17.4868 22.9055 15.4825L22.9058 15.4818C23.0007 15.2532 23.0256 15.0015 22.9774 14.7587C22.9291 14.5159 22.8099 14.2929 22.6348 14.118C22.4597 13.9431 22.2366 13.8241 21.9937 13.7761C21.7509 13.7281 21.4993 13.7533 21.2708 13.8484L21.2707 13.8485C20.2346 14.2801 19.1231 14.5016 18.0007 14.5H18C15.7457 14.5 13.5837 13.6045 11.9896 12.0104C10.3955 10.4163 9.5 8.25433 9.5 5.99999L9.5 5.99927C9.49838 4.8769 9.71983 3.76541 10.1515 2.72938C10.2468 2.50072 10.2721 2.24888 10.224 2.00584C10.1759 1.76281 10.0567 1.53954 9.8815 1.36438Z"/>
      </svg>
      <svg class="main" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <g>
          <path class="outline"
            d="M9.8815 1.36438L9.88141 1.36429C9.70639 1.18942 9.48342 1.07041 9.24073 1.02235C8.99803 0.974286 8.74653 0.999323 8.51808 1.09429L8.51753 1.09452C4.54484 2.75146 1.75 6.6732 1.75 11.25C1.75 17.3262 6.67489 22.25 12.75 22.25C14.9217 22.2501 17.0448 21.6075 18.852 20.4032C20.6591 19.1989 22.0695 17.4868 22.9055 15.4825L22.9058 15.4818C23.0007 15.2532 23.0256 15.0015 22.9774 14.7587C22.9291 14.5159 22.8099 14.2929 22.6348 14.118C22.4597 13.9431 22.2366 13.8241 21.9937 13.7761C21.7509 13.7281 21.4993 13.7533 21.2708 13.8484L21.2707 13.8485C20.2346 14.2801 19.1231 14.5016 18.0007 14.5H18C15.7457 14.5 13.5837 13.6045 11.9896 12.0104C10.3955 10.4163 9.5 8.25433 9.5 5.99999L9.5 5.99927C9.49838 4.8769 9.71983 3.76541 10.1515 2.72938C10.2468 2.50072 10.2721 2.24888 10.224 2.00584C10.1759 1.76281 10.0567 1.53954 9.8815 1.36438Z"
            fill="black" stroke="black" stroke-width="2"/>
          <path mask="url(#lga-fade)" class="outline-shadow" filter="url(#lga-outer-shadow)"
            d="M9.8815 1.36438L9.88141 1.36429C9.70639 1.18942 9.48342 1.07041 9.24073 1.02235C8.99803 0.974286 8.74653 0.999323 8.51808 1.09429L8.51753 1.09452C4.54484 2.75146 1.75 6.6732 1.75 11.25C1.75 17.3262 6.67489 22.25 12.75 22.25C14.9217 22.2501 17.0448 21.6075 18.852 20.4032C20.6591 19.1989 22.0695 17.4868 22.9055 15.4825L22.9058 15.4818C23.0007 15.2532 23.0256 15.0015 22.9774 14.7587C22.9291 14.5159 22.8099 14.2929 22.6348 14.118C22.4597 13.9431 22.2366 13.8241 21.9937 13.7761C21.7509 13.7281 21.4993 13.7533 21.2708 13.8484L21.2707 13.8485C20.2346 14.2801 19.1231 14.5016 18.0007 14.5H18C15.7457 14.5 13.5837 13.6045 11.9896 12.0104C10.3955 10.4163 9.5 8.25433 9.5 5.99999L9.5 5.99927C9.49838 4.8769 9.71983 3.76541 10.1515 2.72938C10.2468 2.50072 10.2721 2.24888 10.224 2.00584C10.1759 1.76281 10.0567 1.53954 9.8815 1.36438Z"
            fill="black" stroke="black" stroke-width="2"/>
        </g>
        <path class="trail" stroke="#2CC6FE" stroke-linecap="round"
          d="M9.8815 1.36438L9.88141 1.36429C9.70639 1.18942 9.48342 1.07041 9.24073 1.02235C8.99803 0.974286 8.74653 0.999323 8.51808 1.09429L8.51753 1.09452C4.54484 2.75146 1.75 6.6732 1.75 11.25C1.75 17.3262 6.67489 22.25 12.75 22.25C14.9217 22.2501 17.0448 21.6075 18.852 20.4032C20.6591 19.1989 22.0695 17.4868 22.9055 15.4825L22.9058 15.4818C23.0007 15.2532 23.0256 15.0015 22.9774 14.7587C22.9291 14.5159 22.8099 14.2929 22.6348 14.118C22.4597 13.9431 22.2366 13.8241 21.9937 13.7761C21.7509 13.7281 21.4993 13.7533 21.2708 13.8484L21.2707 13.8485C20.2346 14.2801 19.1231 14.5016 18.0007 14.5H18C15.7457 14.5 13.5837 13.6045 11.9896 12.0104C10.3955 10.4163 9.5 8.25433 9.5 5.99999L9.5 5.99927C9.49838 4.8769 9.71983 3.76541 10.1515 2.72938C10.2468 2.50072 10.2721 2.24888 10.224 2.00584C10.1759 1.76281 10.0567 1.53954 9.8815 1.36438Z"/>
        <g class="inner">
          <path class="inner-face" fill-rule="evenodd" clip-rule="evenodd"
            d="M9.528 1.71799C9.63312 1.82308 9.70465 1.95704 9.73349 2.10286C9.76234 2.24868 9.7472 2.39979 9.69 2.53699C9.23282 3.6342 8.99828 4.81134 9 5.99999C9 8.38694 9.94821 10.6761 11.636 12.3639C13.3239 14.0518 15.6131 15 18 15C19.1886 15.0017 20.3658 14.7672 21.463 14.31C21.6001 14.2529 21.7511 14.2378 21.8968 14.2666C22.0425 14.2954 22.1763 14.3668 22.2814 14.4717C22.3865 14.5767 22.458 14.7105 22.487 14.8562C22.5159 15.0018 22.501 15.1528 22.444 15.29C21.646 17.2032 20.2997 18.8376 18.5747 19.9871C16.8496 21.1367 14.823 21.7501 12.75 21.75C6.951 21.75 2.25 17.05 2.25 11.25C2.25 6.88199 4.917 3.13799 8.71 1.55599C8.84707 1.49901 8.99797 1.48399 9.14359 1.51282C9.28921 1.54166 9.42299 1.61307 9.528 1.71799Z"/>
          <path mask="url(#lga-inner-fade)" class="inner-bg" fill-rule="evenodd" clip-rule="evenodd"
            d="M9.528 1.71799C9.63312 1.82308 9.70465 1.95704 9.73349 2.10286C9.76234 2.24868 9.7472 2.39979 9.69 2.53699C9.23282 3.6342 8.99828 4.81134 9 5.99999C9 8.38694 9.94821 10.6761 11.636 12.3639C13.3239 14.0518 15.6131 15 18 15C19.1886 15.0017 20.3658 14.7672 21.463 14.31C21.6001 14.2529 21.7511 14.2378 21.8968 14.2666C22.0425 14.2954 22.1763 14.3668 22.2814 14.4717C22.3865 14.5767 22.458 14.7105 22.487 14.8562C22.5159 15.0018 22.501 15.1528 22.444 15.29C21.646 17.2032 20.2997 18.8376 18.5747 19.9871C16.8496 21.1367 14.823 21.7501 12.75 21.75C6.951 21.75 2.25 17.05 2.25 11.25C2.25 6.88199 4.917 3.13799 8.71 1.55599C8.84707 1.49901 8.99797 1.48399 9.14359 1.51282C9.28921 1.54166 9.42299 1.61307 9.528 1.71799Z"/>
          <g class="inner-shadow" filter="url(#lga-inner-shadow)" mask="url(#lga-fade)">
            <path fill-rule="evenodd" clip-rule="evenodd"
              d="M9.528 1.71799C9.63312 1.82308 9.70465 1.95704 9.73349 2.10286C9.76234 2.24868 9.7472 2.39979 9.69 2.53699C9.23282 3.6342 8.99828 4.81134 9 5.99999C9 8.38694 9.94821 10.6761 11.636 12.3639C13.3239 14.0518 15.6131 15 18 15C19.1886 15.0017 20.3658 14.7672 21.463 14.31C21.6001 14.2529 21.7511 14.2378 21.8968 14.2666C22.0425 14.2954 22.1763 14.3668 22.2814 14.4717C22.3865 14.5767 22.458 14.7105 22.487 14.8562C22.5159 15.0018 22.501 15.1528 22.444 15.29C21.646 17.2032 20.2997 18.8376 18.5747 19.9871C16.8496 21.1367 14.823 21.7501 12.75 21.75C6.951 21.75 2.25 17.05 2.25 11.25C2.25 6.88199 4.917 3.13799 8.71 1.55599C8.84707 1.49901 8.99797 1.48399 9.14359 1.51282C9.28921 1.54166 9.42299 1.61307 9.528 1.71799Z"
              fill="hsl(0 0% 10% / .01)"/>
          </g>
        </g>
        <defs>
          <filter id="lga-inner-shadow" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
            <feFlood flood-opacity="0" result="BackgroundImageFix"/>
            <feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"/>
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha"/>
            <feOffset dx="0.4" dy="0.5"/>
            <feGaussianBlur stdDeviation="0.1"/>
            <feComposite in2="hardAlpha" operator="arithmetic" k2="-1" k3="1"/>
            <feColorMatrix type="matrix" values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0"/>
            <feBlend mode="normal" in2="shape" result="effect1_innerShadow"/>
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha"/>
            <feOffset dx="0.3" dy="-0.5"/>
            <feGaussianBlur stdDeviation="0.1"/>
            <feComposite in2="hardAlpha" operator="arithmetic" k2="-1" k3="1"/>
            <feColorMatrix type="matrix" values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0"/>
            <feBlend mode="normal" in2="effect1_innerShadow" result="effect2_innerShadow"/>
          </filter>
          <linearGradient id="lga-fade-gradient" x1="0%" y1="0%" x2="100%" y2="0%" gradientTransform="rotate(45)" gradientUnits="userSpaceOnUse">
            <stop offset="0.45" stop-color="white" stop-opacity="0"/>
            <stop offset="0.75" stop-color="white" stop-opacity="0.75"/>
            <stop offset="0.95" stop-color="white" stop-opacity="0.5"/>
            <stop offset="1" stop-color="white" stop-opacity="0.35"/>
          </linearGradient>
          <linearGradient id="lga-inner-fade-gradient" x1="0%" y1="0%" x2="100%" y2="0%" gradientTransform="rotate(45)" gradientUnits="userSpaceOnUse">
            <stop offset="0" stop-color="transparent" stop-opacity="0"/>
            <stop offset="0.75" stop-color="white" stop-opacity="1"/>
          </linearGradient>
          <mask id="lga-fade">
            <rect width="100%" height="100%" fill="url(#lga-fade-gradient)"/>
          </mask>
          <mask id="lga-inner-fade">
            <rect width="100%" height="100%" fill="url(#lga-inner-fade-gradient)"/>
          </mask>
          <filter id="lga-outer-shadow" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
            <feFlood flood-opacity="0" result="BackgroundImageFix"/>
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha"/>
            <feOffset dy="0.5" dx="-0.05"/>
            <feGaussianBlur stdDeviation="0.25"/>
            <feComposite in2="hardAlpha" operator="out"/>
            <feColorMatrix type="matrix" values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0"/>
            <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow"/>
            <feBlend mode="normal" in="SourceGraphic" in2="effect1_dropShadow" result="shape"/>
          </filter>
        </defs>
      </svg>
    </div>
    <span class="sr-only">Toggle Theme</span>
  </button>
</div>

<!-- custom platinum cursor (populated + moved by JS) -->
<div class="lga-cursor" id="lga-cursor"></div>

<!-- custom Generating loader overlay -->
<div id="lga-loader-overlay">
  <div class="loader-wrapper">
    <span class="loader-letter">G</span>
    <span class="loader-letter">e</span>
    <span class="loader-letter">n</span>
    <span class="loader-letter">e</span>
    <span class="loader-letter">r</span>
    <span class="loader-letter">a</span>
    <span class="loader-letter">t</span>
    <span class="loader-letter">i</span>
    <span class="loader-letter">n</span>
    <span class="loader-letter">g</span>
    <div class="loader"></div>
  </div>
</div>
"""


# ============================================================
# JAVASCRIPT
# ============================================================
SETUP_JS = r"""
() => {
  const CUR_BLACK = "<svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><defs><linearGradient id='lgaCurB' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='#525b66'/><stop offset='40%' stop-color='#21252b'/><stop offset='70%' stop-color='#0d0e11'/><stop offset='100%' stop-color='#3a414b'/></linearGradient></defs><path d='M 3,2 L 20,12 L 12,14 L 8,22 Z' fill='url(#lgaCurB)' stroke='#8a95a5' stroke-width='1.2' stroke-linejoin='round'/></svg>";
  const CUR_WHITE = "<svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><defs><linearGradient id='lgaCurW' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='#ffffff'/><stop offset='45%' stop-color='#e2e8f0'/><stop offset='70%' stop-color='#94a3b8'/><stop offset='100%' stop-color='#ffffff'/></linearGradient></defs><path d='M 3,2 L 20,12 L 12,14 L 8,22 Z' fill='url(#lgaCurW)' stroke='#ffffff' stroke-width='1.2' stroke-linejoin='round'/></svg>";

  function ensureCursor() {
    let c = document.getElementById('lga-cursor');
    if (!c) {
      c = document.createElement('div');
      c.id = 'lga-cursor';
      c.className = 'lga-cursor';
      document.body.appendChild(c);
    } else if (c.parentElement !== document.body) {
      document.body.appendChild(c);
    }
    return c;
  }

  function setCursor(theme) {
    const c = ensureCursor();
    if (c.dataset.theme !== theme || !c.firstChild) {
      c.innerHTML = (theme === 'light') ? CUR_WHITE : CUR_BLACK;
      c.dataset.theme = theme;
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btn = document.getElementById('lga-theme-toggle');
    if (btn) btn.setAttribute('aria-pressed', theme === 'light' ? 'true' : 'false');
    setCursor(theme);
  }

  const ICONS = {
    'Text':         '<svg class="tab-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path class="f" d="M1 2 L15 2 L15 4 L9 4 L9 14 L7 14 L7 4 L1 4 Z"/></svg>',
    'PDF Document': '<svg class="tab-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path class="f" fill-rule="evenodd" d="M2 1 L9 1 L14 6 L14 15 L2 15 Z M4 9 L12 9 L12 10.5 L4 10.5 Z M4 12 L10 12 L10 13.5 L4 13.5 Z"/></svg>',
    'Image':        '<svg class="tab-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><rect class="s" x="1.5" y="2.5" width="13" height="11" rx="1.5"/><circle class="f" cx="4.5" cy="6" r="1.5"/><path class="f" d="M2 13.5 L6 9 L9 11 L12 8 L14 10.5 L14 13.5 Z"/></svg>',
    'Audio':        '<svg class="tab-icon" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><rect class="f" x="0" y="6" width="2" height="4" rx="1"/><rect class="f" x="3.5" y="4" width="2" height="8" rx="1"/><rect class="f" x="7" y="2" width="2" height="12" rx="1"/><rect class="f" x="10.5" y="4" width="2" height="8" rx="1"/><rect class="f" x="14" y="6" width="2" height="4" rx="1"/></svg>'
  };

  function injectIcons() {
    document.querySelectorAll('div[role="tablist"] button').forEach(function (b) {
      if (b.querySelector('.tab-inner')) return;
      const label = (b.textContent || '').trim();
      const svg = ICONS[label];
      if (!svg) return;
      b.innerHTML = '<span class="tab-inner">' + svg + '<span>' + label + '</span></span>';
    });
  }

  function setupDock() {
    document.querySelectorAll('div[role="tablist"]').forEach(function (list) {
      if (list.dataset.dock === '1') return;
      list.dataset.dock = '1';
      const MAX = 0.26;
      const DIST = 130;
      list.addEventListener('mousemove', function (e) {
        list.querySelectorAll('button').forEach(function (b) {
          const r = b.getBoundingClientRect();
          const center = r.left + r.width / 2;
          const d = Math.abs(e.clientX - center);
          const t = Math.max(0, 1 - d / DIST);
          b.style.transform = 'scale(' + (1 + MAX * t).toFixed(3) + ')';
          b.style.zIndex = String(100 + Math.round(t * 100));
        });
      });
      list.addEventListener('mouseleave', function () {
        list.querySelectorAll('button').forEach(function (b) {
          b.style.transform = '';
          b.style.zIndex = '';
        });
      });
    });
  }

  function setupCursorFollow() {
    ensureCursor();
    if (window.__lgaCursorBound) return;
    window.__lgaCursorBound = true;
    document.addEventListener('mousemove', function (e) {
      const c = ensureCursor();
      c.style.transform = 'translate(' + (e.clientX - 3) + 'px,' + (e.clientY - 2) + 'px)';
    }, { passive: true });
    document.addEventListener('mouseleave', function () {
      const c = document.getElementById('lga-cursor'); if (c) c.style.opacity = '0';
    });
    document.addEventListener('mouseenter', function () {
      const c = document.getElementById('lga-cursor'); if (c) c.style.opacity = '1';
    });
  }

  function showLoader() {
    const o = document.getElementById('lga-loader-overlay');
    if (o) o.classList.add('show');
    clearTimeout(window.__lgaLoaderTimer);
    window.__lgaLoaderTimer = setTimeout(hideLoader, 90000);
  }
  function hideLoader() {
    const o = document.getElementById('lga-loader-overlay');
    if (o) o.classList.remove('show');
  }
  function setupLoader() {
    document.querySelectorAll('.glass-btn button').forEach(function (b) {
      if (b.dataset.lgaLoader === '1') return;
      b.dataset.lgaLoader = '1';
      b.addEventListener('click', showLoader);
    });
    if (!window.__lgaOutObs) {
      const outs = document.querySelectorAll('.output-box textarea');
      const obs = new MutationObserver(hideLoader);
      outs.forEach(function (t) {
        obs.observe(t, { attributes: true, attributeFilter: ['value'] });
        t.addEventListener('input', hideLoader);
      });
      window.__lgaOutObs = obs;
      window.__lgaOutText = {};
      setInterval(function () {
        document.querySelectorAll('.output-box textarea').forEach(function (t, i) {
          if (window.__lgaOutText[i] === undefined) window.__lgaOutText[i] = t.value;
          else if (window.__lgaOutText[i] !== t.value) { window.__lgaOutText[i] = t.value; hideLoader(); }
        });
      }, 400);
    }
  }

  function ensureOrbs() {
    const cont = document.querySelector('.gradio-container');
    if (!cont || cont.querySelector('#lga-orb-field')) return;
    const field = document.createElement('div');
    field.id = 'lga-orb-field';
    field.innerHTML =
      '<div class="orb orb-1"></div><div class="orb orb-2"></div>' +
      '<div class="orb orb-3"></div><div class="orb orb-4"></div>' +
      '<div class="orb orb-5"></div><div class="orb orb-6"></div>' +
      '<div class="orb orb-7"></div>';
    cont.insertBefore(field, cont.firstChild);
  }

  function setup() {
    const saved = localStorage.getItem('lga-theme') || 'dark';
    applyTheme(saved);

    const btn = document.getElementById('lga-theme-toggle');
    if (btn && btn.dataset.bound !== '1') {
      btn.dataset.bound = '1';
      btn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        const cur  = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = (cur === 'light') ? 'dark' : 'light';
        applyTheme(next);
        localStorage.setItem('lga-theme', next);
      });
    }

    injectIcons();
    setupDock();
    setupCursorFollow();
    setupLoader();
    ensureOrbs();
  }

  setup();
  [100, 400, 1000, 2000, 3500].forEach(function (t) { setTimeout(setup, t); });
}
"""


# ============================================================
# UI
# ============================================================
LOADER_SHOW_JS = "() => { const o = document.getElementById('lga-loader-overlay'); if (o) o.classList.add('show'); }"
LOADER_HIDE_JS = "() => { const o = document.getElementById('lga-loader-overlay'); if (o) o.classList.remove('show'); }"

with gr.Blocks(
    title="Mini AI Assistant — Week 4",
    css=LIQUID_GLASS_CSS,
    theme=gr.themes.Base()
) as demo:

    gr.HTML(THEME_TOGGLE_HTML)

    gr.HTML("""
    <div class="header-box" style="max-width: 800px; margin: 0 auto 22px auto;">
        <h1>Orbit AI Assistant</h1>
        <div class="header-sub">
            <p class="agent-subtitle" style="font-size: 1.2rem;">Agent-powered tool selection</p>
            <p class="agent-description" style="font-size: 0.85rem;">
                The agent reads your input and automatically decides which tool to use.
            </p>
        </div>
    </div>
    """)

    with gr.Tab("Text"):
        with gr.Group(elem_classes=["glass-card"]):
            gr.HTML('<span class="tool-label">text_tool</span>')
            gr.Markdown(
                "Type any sentence. The agent will **classify** it (Sports / Politics / "
                "Technology / Health / Entertainment) and **extract** key information "
                "(Person, Organization, Location, Date, Event)."
            )
            text_input = gr.Textbox(
                label="Your text",
                placeholder='e.g. "Mohamed Salah scored 2 goals against Manchester City on December 1st."',
                lines=3,
            )
            text_btn = gr.Button("Analyze Text", variant="primary", elem_classes=["glass-btn"])
            text_output = gr.Textbox(label="Agent Response", lines=15, max_lines=60, elem_classes=["output-box"])
            text_btn.click(fn=None, js=LOADER_SHOW_JS).then(
                fn=handle_text, inputs=text_input, outputs=text_output).then(
                fn=None, js=LOADER_HIDE_JS)

    with gr.Tab("PDF Document"):
        with gr.Group(elem_classes=["glass-card"]):
            gr.HTML('<span class="tool-label">document_tool</span>')
            gr.Markdown(
                "Upload a PDF (invoice, report, form). The agent will extract and "
                "summarize the key information from it."
            )
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_btn = gr.Button("Extract from PDF", variant="primary", elem_classes=["glass-btn"])
            pdf_output = gr.Textbox(label="Agent Response", lines=20, max_lines=80, elem_classes=["output-box"])
            pdf_btn.click(fn=None, js=LOADER_SHOW_JS).then(
                fn=handle_pdf, inputs=pdf_input, outputs=pdf_output).then(
                fn=None, js=LOADER_HIDE_JS)

    with gr.Tab("Image"):
        with gr.Group(elem_classes=["glass-card"]):
            gr.HTML('<span class="tool-label">image_tool</span>')
            gr.Markdown("Upload a chest X-ray. The model classifies it as Normal, Pneumonia, COVID-19, or Tuberculosis.")
            image_input = gr.Image(label="Upload Image", type="filepath")
            image_btn = gr.Button("Classify Image", variant="primary", elem_classes=["glass-btn"])
            image_output = gr.Textbox(label="Agent Response", lines=4, max_lines=10, elem_classes=["output-box"])
            image_btn.click(fn=None, js=LOADER_SHOW_JS).then(
                fn=handle_image, inputs=image_input, outputs=image_output).then(
                fn=None, js=LOADER_HIDE_JS)

    with gr.Tab("Audio"):
        with gr.Group(elem_classes=["glass-card"]):
            gr.HTML('<span class="tool-label">audio_tool</span>')
            gr.Markdown(
                "Upload an audio file (.mp3, .wav, .m4a). Groq's Whisper model will "
                "transcribe the speech into text."
            )
            audio_input = gr.Audio(
                label="Upload or Record Audio",
                type="filepath",
                sources=["microphone", "upload"],
            )
            audio_btn = gr.Button("Transcribe Audio", variant="primary", elem_classes=["glass-btn"])
            audio_output = gr.Textbox(label="Agent Response", lines=8, max_lines=40, elem_classes=["output-box"])
            audio_btn.click(fn=None, js=LOADER_SHOW_JS).then(
                fn=handle_audio, inputs=audio_input, outputs=audio_output).then(
                fn=None, js=LOADER_HIDE_JS)

    demo.load(None, None, None, js=SETUP_JS)


# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.launch()