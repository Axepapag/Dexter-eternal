import os
import threading
import asyncio
import edge_tts
import tempfile
import ctypes
import time
from pathlib import Path

# Windows-specific MCI playback
def play_mp3_native(path):
    """Play MP3 using Windows MCI (Media Control Interface)."""
    try:
        # Use short path names to avoid issues with spaces
        # But for temp files, we just wrap in quotes
        path = os.path.abspath(path)
        
        # MCI commands
        ctypes.windll.winmm.mciSendStringW(f'open "{path}" type mpegvideo alias dexter_voice', None, 0, 0)
        ctypes.windll.winmm.mciSendStringW('play dexter_voice wait', None, 0, 0)
        ctypes.windll.winmm.mciSendStringW('close dexter_voice', None, 0, 0)
        return True
    except Exception as e:
        print(f"[Native Playback Error] {e}")
        return False

class Speaker:
    def __init__(self, voice="en-US-GuyNeural", rate="+50%"):
        self.voice = voice
        self.rate = rate
        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _synthesize_and_play(self, text):
        """Internal async method to handle TTS workflow."""
        # Use a stable temp location
        env_temp = os.getenv("DEXTER_TEMP_DIR")
        if env_temp:
            temp_dir = Path(env_temp)
        else:
            repo_root = Path(__file__).resolve().parent.parent
            temp_dir = repo_root / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=temp_dir) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # 1. Synthesize
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(temp_path)
            
            # 2. Play using native Windows MCI
            # This is synchronous but runs in the background thread of the loop
            play_mp3_native(temp_path)
            
        except Exception as e:
            print(f"[Speaker Error] {e}")
        finally:
            # 3. Cleanup
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def speak(self, text):
        """Speaks the text in a non-blocking way."""
        if not text: return
        # Schedule the task on the background event loop
        asyncio.run_coroutine_threadsafe(self._synthesize_and_play(text), self.loop)

# Global speaker instance
_speaker = Speaker()

def speak_out_loud(text: str):
    """Clean interface for Dexter to communicate directly with Jeffrey."""
    if not text: return
    # Filter out text that shouldn't be spoken
    if text.startswith("["): return
    
    print(f"\n[Dexter Speaking] \"{text}\"")
    _speaker.speak(text)
