import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")  # You can change this to "tiny", "small", "medium", or "large" based on your needs

model = None

def load_model():
    global model
    if model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        model = whisper.load_model(WHISPER_MODEL)
        print("Model loaded successfully.")
    return model 

def transcribe_chunk(chunk_path: str,translate: bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    return result["text"]

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""
    for i ,chunk in enumerate(chunks):
        text = transcribe_chunk(chunk, translate=translate)
        full_transcript += text + " "
        print(f"Transcription completed for chunk {i+1}.")
    return full_transcript