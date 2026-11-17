# Nova Voice

A Streamlit voice assistant with English, Urdu, and mixed-language transcription,
conversation context, editable transcripts, and saved audio in four voices.

The interface follows the user's published Nova design, using its original
nebula and planet artwork, Orbitron/Sora typography, glass panel, cyan/magenta
gradients, and animated stars and orbits. See `assets/REFERENCE.md` for provenance.
Dark mode uses the reference colors; Light and System modes are also available.
The sidebar includes an animation toggle and respects reduced-motion settings.

## Run

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Paste an OpenAI API key in the sidebar, or copy `.env.example` to `.env` and
set `OPENAI_API_KEY`. Sidebar input takes priority. The key is used on the server
and is never written to a file by the app. An API account with available credits
and access to the configured models is required.

Allow microphone access, record a question, then press Stop. The app transcribes
it, displays the answer, and generates an AI voice reply. Select a playback voice
on an individual reply and click **Try another voice** to generate it. Saved audio
uses the browser player's play, pause, and seek controls without new API calls.
The sidebar voice controls new replies. Appearance follows the browser's system
preference in System mode, including preference changes while the app is open.
Newly generated audio requests automatic playback once. Later reruns and replay
use saved audio without new API calls. Browser autoplay restrictions may require
allowing sound for the site or using the visible player. You can also type into
the message composer or use a starter mission to begin a conversation.

Each failed stage has a retry action. Text remains available if speech fails.
Editing an old transcript regenerates that reply using only preceding turns;
existing later replies remain as originally generated. History and audio are
held in Streamlit session memory and may be lost on reload or disconnect.
Clear conversation resets the recorder and removes session history.

Microphone capture requires localhost or HTTPS. Realtime speech also requires
outbound WebSocket access to `api.openai.com`. Speech is model-generated and may
occasionally differ from the written text; check names, numbers, and pronunciation.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Also check one English, one Urdu, and one mixed-language recording; a contextual
follow-up; transcript corrections; replaying older audio; switching voices and
switching back; all three appearance modes; and retrying after a network failure.
These live checks require your own API key and microphone.
