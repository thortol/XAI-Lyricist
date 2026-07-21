import os

from dotenv import load_dotenv
from fastapi import Form, HTTPException, Request, APIRouter
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Dict

from midi_to_wav import get_syllables, convert
from database import Database

router = APIRouter(prefix="/prototype", tags=["prototype"])
load_dotenv()  

midi_path = os.getenv("MIDI_FILES_PATH", "midi_files")
midi_files = {
    "Find My Way Back Home": os.path.join(midi_path, "find_my_way_back_home.mid"),
    "Imagine": os.path.join(midi_path, "imagine.mid"),
    "Million Reasons": os.path.join(midi_path, "million_reasons.mid"),
    "Set Fire to the Rain": os.path.join(midi_path, "set_fire_to_the_rain.mid"),
    "Stay With Me": os.path.join(midi_path, "stay_with_me.mid")
} # map it to the mid file, hardcoded to prevent attacks

db_audio_files = {
    "Find My Way Back Home": "find my way back home.wav",
    "Imagine": "imagine.mp3",
    "Million Reasons": "million reasons.wav",
    "Set Fire to the Rain": "set fire to the rain.wav",
    "Stay With Me": "stay with me.wav"
}

database = Database(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

question_bank = [
    "Who is this song about?", 
    "What happened?", 
    "How did it make you feel?", 
    "Where did it happen?", 
    "What message do you want people to remember?",
    "What kind of mood should the lyrics have?",
    "Is there a favourite word or phrase you'd like included?"
]

openai_client = OpenAI(api_key=os.getenv("OPENAPI_KEY"))

class Message(BaseModel):
    role: str
    content: str

class SongWritingRequest(BaseModel):
    song_title: str
    chat_history: list[Message]

@router.post("/song-writing")
async def song_writing(
    body: SongWritingRequest,
    request: Request
) -> Dict[str, str]:
    
    form = await request.form()
    print("ALL FIELDS RECEIVED:", dict(form))
    print(body.song_title, body.chat_history)

    # if request.headers.get("Authorization") == None or "Bearer " not in request.headers.get("Authorization"):
    #     raise HTTPException(status_code=400, detail="auth token is required")
    
    token = request.headers.get("Authorization").replace("Bearer ", "")

    # if not database.validate_user(token):
    #     raise HTTPException(status_code=400, detail="valid user auth token is required")

    if body.song_title not in midi_files:
        raise HTTPException(status_code=400, detail="valid song title is required")
    
    
    possible_questions = question_bank[:]
    asked_questions = 0
    convo = ""
    for chat in body.chat_history:
        convo += chat.role + " - " + chat.content + "\n"
        if chat.role == "AI" and chat.content in possible_questions:
            possible_questions.remove(chat.content)
            asked_questions += 1

    questions = ""
    i = 1
    for question in possible_questions:
        questions += str(i) + " - " + question + "\n"
        i += 1
    if asked_questions <= 2:
        response = openai_client.responses.create(
        model="gpt-4o-mini",
        input=f"""You are a songwriting AI writing lyrics for stroke patients. Currently your task is to find out more about the patient. 
Here is the conversation so far, decide on the next question to be asked. Make sure the next question has coherent sense. 
For example, if the prompt is about a person, choose a question that further prompts about that person.
However if the prompt does not mention a person, do not ask who the song is about.
Return the output as a single integer. Return 0 if there is no need to ask further questions anymore or there is an unexpected error.

Conversation:
{convo}
Available questions:
{questions}
Return only an integer or return 0 if there is no need to ask further questions anymore or the prompt has no relevance to the given questions.
""")
        i = int(response.output_text) - 1
    else:
        i = -1
 
    if i != -1: 
        return {"reply": possible_questions[i]}

    response = openai_client.responses.create(
    model="gpt-4o-mini",
    input=f"""You are a songwriting AI writing lyrics for stroke patients. Currently your task is to write a song for the patient. 
Here is the conversation so far, only return the lyrics related to the song you are trying to write.
You are trying to write a song for the melody of {body.song_title}. 
The song that you are trying to write should have {get_syllables(midi_files[body.song_title])} syllables

Conversation:
{convo}

Return only the lyrics of the song. End each line with a single \n, and each section with \n\n. 
There should be 3 sections, intro, verse and chorus. 
Do not use words with ` like 'we`ll'.
""")
    # if database.validate_user(token):
    #     wav_data, karaoke_timing = convert(midi_files[body.song_title], response.output_text)
    #     song_id = database.insert_song(wav_data, token, body.song_title, "temporary prompt", response.output_text, karaoke_timing, {}, db_audio_files[body.song_title], False)


    return {"lyrics": response.output_text}