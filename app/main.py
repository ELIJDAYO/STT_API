from typing import List
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from labels import tagalog_labels
from dictionaries import dictionaryTagalogRanges, numericals
from expected_labels import tagalog_expected_labels
from fuzzywuzzy import process
from typing import List, Optional
from pydantic import BaseModel
import os
import json
import traceback
import re
import inflect
from collections import defaultdict

# Sample Dictionary, make it a file when finalized.
# numberTagalog_MTO1 = {
#     # QUESTION 1
#     "Question 1":
#         {
#             "one" : "1",
#             "isa" : "1",
#             "isa lang": "1",
#             "two" : "2",
#             "dalawa" : "2",
#             "dalawa lang" : "2"
#         }
#     , 

#     "Question 2": 
#         {
#             # sample with many synonyms
#             "mga dalawang araw": "2-3 na araw",
#             "dalawang araw": "2-3 na araw",
#             "tatlong araw": "2-3 na araw", 
#             "mga tatlong araw": "2-3 na araw", 
#             "dalawa": "2-3 na araw",
#             "tatlo": "2-3 na araw",
        
#             # sample with few synonyms
#             "apat na araw": "4-6 na araw",
#             "mga anim": "4-6 na araw"
#         }
#     ,

#     "Question 3": 
#         {
#             "oo": "Oo",
#             "yes": "Oo",
#             "oh": "Oo",

#             "hindi": "Hindi",
#             "no": "Hindi",
#             "dili": "Hindi"
#         }
# }


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
    trans_str, match, score = map_transcripts(transcription, tagalog_labels, labels)

    response = {
        "transcription": trans_str,
        "match": match,
        "score": score
    }
    return JSONResponse(content=response)


class TranscriptionRequest(BaseModel):
    text: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = None


# Define your transcription function here
def show_transcription(model, processor, audio_file):
    try:
        audio_input, sampling_rate = librosa.load(audio_file, sr=16000)
        inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0]  # Return a single string
    except Exception as e:
        # print(f"Error during transcription: {e}")
        # traceback.print_exc()
        return None


def map_transcripts(transcript, labels, expected_labels=None):
    trans_str = transcript.strip()  # Ensure there are no leading/trailing spaces
    trans_length = len(trans_str)

    # Filter labels based on length difference
    filtered_labels = [label.strip() for label in labels if abs(len(label) - trans_length) <= 5]

    print(f"Transcription: '{trans_str}' (length: {trans_length})")
    print(f"Filtered labels count: {len(filtered_labels)}")
    print(f"Sample filtered labels: {filtered_labels[:5]}")  # Print the first 5 filtered labels for inspection

    if filtered_labels:
        try:
            from fuzzywuzzy import process  # Importing here to avoid issues if fuzzywuzzy is not installed
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

# ------------------------------------------------------------------------------------------------
# --------------------Accepting Choices from FilBis-----------------------------------------------
# ------------------------------------------------------------------------------------------------


def number_to_text(number):
    return p.number_to_words(number)


def replace_numbers(text, numericals):
    for number, word in numericals.items():
        text = text.replace(number, word)
    return text


value_to_keys = defaultdict(list)


def range_to_text(range_str):
    # Check if the range contains a decimal point
    if '.' in range_str:
        # Split the range string by hyphen
        start, end = range_str.split('-')
        start = start.strip()
        end = end.strip()

        # Convert both parts of the range to text
        start_text = number_to_text(start)
        end_text = number_to_text(end)

        return f"{start_text} to {end_text}"
    else:
        return range_str  # Return the original string if no decimal point


def process_list(file_contents):
    for i, s in enumerate(file_contents):
        if '-' in s and re.search(r'\d', s):
            file_contents[i] = range_to_text(s)
    return file_contents


def print_result(mapped_transcripts):
    for original, mapped, score in mapped_transcripts:
        print(f"Original: {original} -> Mapped: {mapped} (Score: {score})")


def map_transcripts2(transcript, choices):
    copy_choices = choices[:]
    processed_choices = process_list(copy_choices)
    updated_choices = [replace_numbers(sentence, numericals) for sentence in processed_choices]

    foundRanges = []
    for item in choices:
        foundRanges.append(value_to_keys[item])

    flattened_list = [item for sublist in foundRanges for item in sublist]
    for item in updated_choices:
        if item not in flattened_list:
            flattened_list.append(item)

    transcript_report = map_transcripts([transcript], flattened_list)
    if transcript_report[0][2] > 30:
        try:
            value = dictionaryTagalogRanges[transcript_report[0][1]]
        except KeyError:
            value = None
    if value is None:
        value = transcript_report[0][1]

    return value


class TranscriptionRequest(BaseModel):
    text: Optional[str] = None
    speaker: Optional[str] = None
    language: Optional[str] = None


def show_transcription2(model, processor, audio_path):
    try:
        # Read audio data using librosa
        audio_input, sampling_rate = librosa.load(audio_path, sr=16000)

        # Process and transcribe audio
        inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0]  # Return a single string
    except Exception as e:
        print(f"Error during transcription: {e}")
        traceback.print_exc()
        return None


@app.post("/transcribe3/")
async def transcribe(
        file: UploadFile = File(None),
        text: Optional[str] = Form(None),
        speaker: str = Form(...),
        language: str = Form(...),
        prompt: Optional[str] = Form(None),
        choices: Optional[str] = Form(None)  # Accept choices as a JSON string
):
    # Log the payload to the console
    print(f"Received payload - Text: {text}, Speaker: {speaker}, Language: {language}, File: {file.filename if file else 'None'}, Choices: {choices}, Prompt: {prompt}")

    response = {}

    # Parse the choices string into a list
    parsed_choices = []
    if choices:
        try:
            parsed_choices = json.loads(choices)
        except json.JSONDecodeError:
            print("Failed to parse choices JSON")
            parsed_choices = []

    if text:
        # Handle text input
        response = {
            "transcription": text,
            "match": "N/A",
            "score": "N/A",
            "choices": parsed_choices
        }

    elif file:
        try:
            # Read the uploaded audio file
            audio_data = await file.read()

            # Ensure the file is in the correct format
            if not file.filename.endswith('.wav'):
                raise ValueError("Unsupported audio format. Please upload a WAV file.")

            # Check if the file has content
            if len(audio_data) == 0:
                raise ValueError("The uploaded WAV file is empty.")

            # Ensure the audio_files directory exists
            os.makedirs("./audio_files", exist_ok=True)

            # Save the audio file to the app directory
            audio_path = f"./audio_files/{file.filename}"
            with open(audio_path, "wb") as buffer:
                buffer.write(audio_data)

            # Perform transcription
            transcription = show_transcription2(model, processor, audio_path)
            if transcription is None:
                response = {"error": "Failed to transcribe audio"}
            else:
                # Assuming tagalog_labels and expected_labels are defined elsewhere
                if parsed_choices:
                    # TODO
                    # Need help in code-switching and synonyms, range values
                    # Please check the map_transcripts and dictionaries.py
                    trans_str, match, score = map_transcripts(transcription, parsed_choices)

                    # Use the line below for debugging
                    # response = {
                    #     "transcription": trans_str,
                    #     "match": match,
                    #     "score": score,
                    #     "choices": parsed_choices
                    # }
                    print(response)

                    response = match

                else:
                    # TODO
                    # This code block is for the open-ended question. It happens when choice array is empty.
                    # We're lacking context such as misspelled letters in Filipino and understanding the prompts when
                    # looking for desired transcription output (e.g. age, weight, school location)
                    # Typing is also strict for some prompts
                    response = transcription
        except Exception as e:
            response = {"error": f"Failed to process audio file: {e}"}
    else:
        response = {"error": "No valid input provided"}

    return response


if __name__ == "__main__":
    import uvicorn

    p = inflect.engine()
    uvicorn.run(app, host="0.0.0.0", port=8000)
