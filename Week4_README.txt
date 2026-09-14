Week 4 - Multimodal AI Agent
==============================
Mohamed Assaad

WHAT THIS PROJECT DOES
-----------------------
A multimodal AI agent built with LangGraph that combines all previous weeks
into one intelligent system with a Liquid Glass Gradio web interface.

The agent receives any input and automatically decides which tool to use:

  Tool 1 - Text Analysis
    Classifies text into a category AND extracts structured information
    (Person, Organization, Location, Date, Event) from any sentence.
    Built on Week 2 work.

  Tool 2 - PDF Analysis
    Reads and answers questions about uploaded PDF documents using
    text extraction and LLM reasoning.

  Tool 3 - Image Classification
    Classifies uploaded images as Cat or Dog using the MobileNetV2
    model trained in Week 3.

  Tool 4 - Audio Transcription
    Transcribes voice notes and audio files to text using Whisper,
    then processes the transcript through the agent.

The agent (LangGraph) routes each input to the correct tool automatically
using LLM decision-making — no hardcoded if/else logic.

FOLDER STRUCTURE
-----------------
Week4_multimodal_agent_proj/
├── multimodal_liquid_glass_ui.py  → main code (agent + UI)
├── requirements.txt               → all required libraries
├── .env                           → API key file (keep this private)
├── cat_dog_model.keras            → trained model from Week 3 (required)
└── venv/                          → virtual environment (created by you)

REQUIREMENT
------------
- Internet connection required (agent brain and audio tool run on Groq servers)
- cat_dog_model.keras must be in the same folder (already included)
- .env file must contain your Groq API key (already set up)

HOW TO RUN
-----------
Run these commands one by one in the terminal:

    cd ..
    cd Week4_multimodal_agent_proj
    dir
    venv\Scripts\activate
    python multimodal_liquid_glass_ui.py

Then open your browser and go to:
    http://127.0.0.1:7860

The Liquid Glass interface will open with four tabs:
Text, PDF, Image, and Audio.

NOTE: If venv does not exist yet, run these instead:
    cd ..
    cd Week4_multimodal_agent_proj
    dir
    python -m venv venv
    venv\Scripts\activate
    pip install python-dotenv langchain-core langchain-groq langgraph groq tensorflow Pillow pypdf gradio
    pip install -r requirements.txt
    python multimodal_liquid_glass_ui.py
    or  python -m gradio multimodal_liquid_glass_ui.py

MODEL & TOOLS USED
-------------------
- Agent brain:    openai/gpt-oss-20b via Groq API
- Text tools:     openai/gpt-oss-20b via Groq API
- Audio (STT):    whisper-large-v3 via Groq API
- Image (local):  MobileNetV2 (TensorFlow/Keras) — cat_dog_model.keras
- UI framework:   Gradio with custom Liquid Glass CSS/JS
- Agent library:  LangGraph (LangChain)

LIMITATIONS
------------
- Requires internet connection for all tools except image classification
- Image tool only classifies Cats and Dogs (trained on PetImages dataset)
- API key is free tier — may have rate limits on heavy usage