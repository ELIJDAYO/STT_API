from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from labels import tagalog_labels
from expected_labels import tagalog_expected_labels
from fuzzywuzzy import process

app = FastAPI()

# Load the Wav2Vec2 model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Read the audio file
    audio_data = await file.read()

    # Save the audio file temporarily
    with open("../temp_audio.wav", "wb") as f:
        f.write(audio_data)

    # Perform speech-to-text
    transcription = show_transcription(model, processor, "../temp_audio.wav")

    # Perform spelling correction
    corrected_transcription = map_transcripts(transcription, tagalog_labels, tagalog_expected_labels)

    return JSONResponse(content={"transcription": corrected_transcription})


# Endpoint to handle audio and list of strings
@app.post("/transcribe2/")
async def transcribe(file: UploadFile = File(...), labels: List[str] = Form(...)):
    # Save the uploaded audio file
    audio_path = f"temp_{file.filename}"
    print(labels)
    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())

    # Perform transcription
    transcription = show_transcription(model, processor, audio_path)

    # Map transcription to expected labels
    trans_str, match, score = map_transcripts(transcription, labels)

    response = {
        "transcription": trans_str,
        "match": match,
        "score": score
    }
    return JSONResponse(content=response)


def show_transcription(model, processor, audio_file):
    audio_input, sampling_rate = librosa.load(audio_file, sr=16000)
    inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]  # Return a single string


def map_transcripts(transcript, labels, expected_labels):
    trans_str = transcript.strip()  # Ensure there are no leading/trailing spaces
    trans_length = len(trans_str)

    # Filter labels based on length difference
    filtered_labels = [label.strip() for label in labels if abs(len(label) - trans_length) <= 5]

    print(f"Transcription: '{trans_str}' (length: {trans_length})")
    print(f"Filtered labels count: {len(filtered_labels)}")
    print(f"Sample filtered labels: {filtered_labels[:5]}")  # Print the first 5 filtered labels for inspection

    if filtered_labels:
        try:
            results = process.extract(trans_str, filtered_labels)
            for match, score in results:
                print(f"Match: {match}, Score: {score}")
            best_match, best_score = results[0][0], results[0][1]
            return trans_str, best_match, best_score
        except Exception as e:
            print(f"Error extracting match for '{trans_str}': {e}")
            return trans_str, None, 0
    else:
        return trans_str, None, 0


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
