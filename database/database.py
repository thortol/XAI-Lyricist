from supabase import create_client

import supabase
import uuid

from datetime import datetime

class Database:
    def __init__(self, supabase_url, supabase_service_key):
        self.supabase = create_client(supabase_url, supabase_service_key)
    
    def insert_song(self, wav_file, token, title, prompt, lyrics, word_timings, metadata, instrumental, flagged):
        user = self.supabase.auth.get_user(token)
        user_id = user.user.id
        filename = f"{uuid.uuid4()}.wav"
        if wav_file != None:
            self.supabase.storage.from_("audio-files").upload(
                path=f"{user_id}/{filename}",
                file=wav_file,
                file_options={"content-type": "audio/wav"}
            )
        else:
            filename = None

        response = self.supabase.table("songs").insert({
            "user_id": user_id,
            "title": title,
            "prompt": prompt,
            "lyrics": lyrics,
            "word_timings": word_timings,
            "midi_url": "",
            "selected_melody_id": "",
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "wav_path": filename,
            "instrumental_path": instrumental,
            "flagged": flagged
        }).select("id").execute()

        return response.data[0]['id']
    
    def validate_user(self, token):
        try:
            user = self.supabase.auth.get_user(token)
            if user.user:
                return True
            return False
        except:
            return False
