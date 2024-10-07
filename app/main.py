from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import whisper
import subprocess
import os
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.llms import OpenAI
from langchain.prompts import PromptTemplate
import whisper
import subprocess


app = FastAPI()

whisper_model = whisper.load_model("base")
process = None  

conversation = ConversationChain(
    prompt=PromptTemplate.from_template("You are an assistant. Respond concisely.")
)

process = subprocess.Popen(["python", "main.py"])



@app.post("/start")
async def start_process():
    global process
    if process is not None and process.poll() is None:
        raise HTTPException(status_code=400, detail="Process already running")
    
    # Start a process (e.g., a long-running LangServe process)
    try:
        process = subprocess.Popen(["process_command_here"])  
        return {"status": "started", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_process():
    global process
    if process is None or process.poll() is not None:
        raise HTTPException(status_code=400, detail="No process running")
    
    try:
        process.terminate()
        process.wait(timeout=10)
        process = None
        return {"status": "stopped"}
    except subprocess.TimeoutExpired:
        process.kill()
        process = None
        return {"status": "force stopped"}

@app.post("/listen/")
async def listen_audio(file: UploadFile = File(...)):

    temp_audio_file = f"temp_audio_{file.filename}"
    with open(temp_audio_file, "wb") as audio:
        audio.write(await file.read())

    result = whisper_model.transcribe(temp_audio_file)
    transcription = result['text']
    
    os.remove(temp_audio_file)
    
    response = conversation.predict(transcription)
    
    return {"transcription": transcription, "response": response}

@app.post("/chain/")
async def listen_audio(file: UploadFile = File(...)):

    if file is None or file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:

        raise HTTPException(status_code=400, detail="Invalid or missing audio file.")

    temp_audio_file = f"temp_audio_{file.filename}"
    
    try:

        with open(temp_audio_file, "wb") as audio:
            audio.write(await file.read())

        # Transcribe the audio using the Whisper model
        result = whisper_model.transcribe(temp_audio_file)
        transcription = result['text']

    except Exception as e:
        # Handle any exception that occurs during file handling or transcription
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up: remove the temporary audio file after processing
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

    # Return the transcription as a response
    return {"transcription": transcription}



# Run the FastAPI server using: uvicorn main:app --reload
