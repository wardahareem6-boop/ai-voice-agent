import asyncio
import base64
import hashlib
import io
import json
import os
from html import escape
from pathlib import Path
import wave
from uuid import uuid4

import streamlit as st
import websockets
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


TRANSCRIPTION_MODEL = "gpt-transcribe"
ANSWER_MODEL = "gpt-5.4-mini"
REALTIME_MODEL = "gpt-realtime-2.1-mini"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
SAMPLE_RATE = 24_000
VOICES = ("marin", "cedar", "coral", "alloy")


class AudioGenerationError(RuntimeError):
    pass


def _init_state() -> None:
    defaults = {
        "conversation": [],
        "processed_recordings": set(),
        "pending_audio": None,
        "audio_errors": {},
        "answer_errors": {},
        "transcription_errors": {},
        "theme_mode": "Dark",
        "api_key_input": "",
        "recorder_version": 0,
        "autoplay_reply": None,
        "motion": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _wav_bytes(pcm_audio: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_audio)
    return output.getvalue()


async def _realtime_audio(api_key: str, text: str, voice: str) -> bytes:
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    chunks = []
    try:
        async with websockets.connect(
            REALTIME_URL,
            additional_headers=headers,
            open_timeout=20,
            close_timeout=10,
        ) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "model": REALTIME_MODEL,
                            "output_modalities": ["audio"],
                            "instructions": (
                                "Read the user's text exactly as written. "
                                "Do not add, omit, translate, or paraphrase anything."
                            ),
                            "audio": {
                                "input": {"turn_detection": None},
                                "output": {
                                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                    "voice": voice,
                                },
                            },
                        },
                    }
                )
            )

            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
                if event.get("type") == "error":
                    raise AudioGenerationError(
                        event.get("error", {}).get("message", "Realtime session failed.")
                    )
                if event.get("type") == "session.updated":
                    break

            await socket.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": text}],
                        },
                    }
                )
            )
            await socket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {"output_modalities": ["audio"]},
                    }
                )
            )

            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=90))
                event_type = event.get("type")
                if event_type == "error":
                    raise AudioGenerationError(
                        event.get("error", {}).get("message", "Realtime audio generation failed.")
                    )
                if event_type == "response.output_audio.delta":
                    chunks.append(base64.b64decode(event["delta"]))
                elif event_type == "response.done":
                    response = event.get("response", {})
                    if response.get("status") != "completed":
                        raise AudioGenerationError(
                            f"Realtime response ended with status "
                            f"{response.get('status', 'unknown')}."
                        )
                    break
    except AudioGenerationError:
        raise
    except (
        websockets.exceptions.WebSocketException,
        OSError,
        asyncio.TimeoutError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        raise AudioGenerationError(f"Could not generate audio: {exc}") from exc

    pcm_audio = b"".join(chunks)
    if not pcm_audio:
        raise AudioGenerationError("The Realtime API returned no audio.")
    return _wav_bytes(pcm_audio)


def _generate_audio(api_key: str, text: str, voice: str) -> bytes:
    async def generate():
        return await asyncio.wait_for(_realtime_audio(api_key, text, voice), timeout=180)
    try:
        return asyncio.run(generate())
    except asyncio.TimeoutError as exc:
        raise AudioGenerationError("Audio generation timed out. Please retry.") from exc


def _history_before(conversation: list[dict], turn: dict) -> list[dict]:
    """Retries and corrections must never use messages from the future."""
    for index, item in enumerate(conversation):
        if item["id"] == turn["id"]:
            return conversation[:index]
    return conversation


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=60, max_retries=1)


def _transcribe(client: OpenAI, audio_bytes: bytes) -> str:
    result = client.audio.transcriptions.create(
        model=TRANSCRIPTION_MODEL,
        file=("question.wav", audio_bytes, "audio/wav"),
    )
    transcript = result.text.strip()
    if not transcript:
        raise ValueError("The transcription service returned an empty transcript.")
    return transcript


def _answer(client: OpenAI, conversation: list[dict], question: str) -> str:
    recent_turns = conversation[-8:]
    messages = []
    for turn in recent_turns:
        if turn.get("question") and turn.get("answer"):
            messages.extend(
                [
                    {"role": "user", "content": turn["question"]},
                    {"role": "assistant", "content": turn["answer"]},
                ]
            )
    messages.append({"role": "user", "content": question})
    response = client.responses.create(
        model=ANSWER_MODEL,
        instructions=(
            "Answer in the same language as the user's latest question. "
            "For mixed-language questions, follow the language the user primarily used "
            "and preserve meaningful names or terms from other languages. "
            "Be helpful, clear, and natural when read aloud."
        ),
        input=messages,
    )
    answer_text = response.output_text.strip()
    if not answer_text:
        raise ValueError("The answer service returned an empty response.")
    return answer_text


def _process_recording(api_key: str, audio_bytes: bytes, recording_id: str) -> None:
    st.session_state.processed_recordings.add(recording_id)
    turn = {
        "id": uuid4().hex,
        "original_transcript": "",
        "question": "",
        "answer": "",
        "audio": {},
        "initial_voice": st.session_state.voice_choice,
        "recording_bytes": audio_bytes,
    }
    st.session_state.conversation.append(turn)
    client = _client(api_key)

    try:
        with st.status("Transcribing your recording…", expanded=True) as status:
            transcript = _transcribe(client, audio_bytes)
            turn["original_transcript"] = transcript
            turn["question"] = transcript
            turn.pop("recording_bytes", None)
            status.update(label="Preparing your answer…")
            turn["answer"] = _answer(client, st.session_state.conversation[:-1], transcript)
            st.session_state.pending_audio = {
                "turn_id": turn["id"],
                "voice": st.session_state.voice_choice,
            }
            status.update(label="Your answer is ready.", state="complete", expanded=False)
    except OpenAIError as exc:
        if not turn["original_transcript"]:
            st.session_state.transcription_errors[turn["id"]] = str(exc)
        else:
            st.session_state.answer_errors[turn["id"]] = str(exc)
    except ValueError as exc:
        if not turn["original_transcript"]:
            st.session_state.transcription_errors[turn["id"]] = str(exc)
        else:
            st.session_state.answer_errors[turn["id"]] = str(exc)


def _submit_corrected_question(api_key: str, turn: dict, corrected: str) -> None:
    corrected = corrected.strip()
    if not corrected:
        st.warning("Enter a question before submitting the correction.")
        return
    turn["question"] = corrected
    turn["answer"] = ""
    turn["audio"] = {}
    st.session_state.transcription_errors.pop(turn["id"], None)
    st.session_state.audio_errors.pop(turn["id"], None)
    st.session_state.answer_errors.pop(turn["id"], None)
    st.session_state.pending_audio = None
    try:
        turn["answer"] = _answer(
            _client(api_key),
            _history_before(st.session_state.conversation, turn),
            corrected,
        )
        st.session_state.pending_audio = {
            "turn_id": turn["id"],
            "voice": st.session_state.voice_choice,
        }
    except (OpenAIError, ValueError) as exc:
        st.session_state.answer_errors[turn["id"]] = str(exc)


def _asset_url(name: str) -> str:
    data = (Path(__file__).parent / "assets" / name).read_bytes()
    mime = "image/jpeg" if name.endswith(".jpg") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _apply_theme(mode: str) -> None:
    dark = """
        --page: oklch(13% .03 270); --text: oklch(95% .02 230);
        --muted: oklch(70% .04 250); --surface: oklch(17% .04 272 / .55);
        --surface-2: oklch(24% .05 275 / .75); --border: oklch(85% .08 220 / .14);
        --accent: oklch(66% .24 350); --accent-2: oklch(82% .14 200);
        --sidebar: oklch(16% .04 270 / .88); --veil: transparent;
        color-scheme: dark;
    """
    light = """
        --page: #e9edf8; --text: #17233f; --muted: #52617e;
        --surface: rgb(244 247 255 / .84); --surface-2: rgb(225 232 249 / .88);
        --border: rgb(76 103 156 / .25); --accent: #b7238d; --accent-2: #087a91;
        --sidebar: rgb(241 245 255 / .94); --veil: rgb(230 238 255 / .66);
        color-scheme: light;
    """
    variables = f":root {{{light if mode == 'Light' else dark}}}"
    if mode == "System":
        variables += f"@media(prefers-color-scheme:light) {{:root {{{light}}}}}"
    css = (Path(__file__).parent / "assets" / "space.css").read_text(encoding="utf-8")
    css = css.replace("__NEBULA__", _asset_url("nebula.jpg"))
    css = css.replace("__PLANET__", _asset_url("planet.png"))
    if not st.session_state.motion:
        css += ".cosmos *, .nova-avatar *, .welcome, [data-testid='stChatMessage'] {animation:none!important}"
    st.markdown(f"<style>{variables}{css}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="cosmos" aria-hidden="true"><div class="nebula"></div>'
        '<div class="stars stars-near"></div><div class="stars stars-far"></div>'
        '<div class="stars stars-mid"></div><div class="vignette"></div>'
        '<div class="space-planet planet-large"></div><div class="space-planet planet-small"></div>'
        '<div class="shooting-star"></div><div class="shooting-star meteor-two"></div>'
        '<div class="shooting-star meteor-three"></div></div>', unsafe_allow_html=True,
    )


def _nova_avatar(large: bool = False) -> str:
    return (
        f'<div class="nova-avatar{" large" if large else ""}" aria-hidden="true">'
        '<span class="orbit-ring"></span><span class="avatar-core">✧</span>'
        '<span class="orbit-dot"></span></div>'
    )


def _process_text(api_key: str, question: str) -> None:
    turn = {"id": uuid4().hex, "original_transcript": question, "question": question,
            "answer": "", "audio": {}, "initial_voice": st.session_state.voice_choice}
    st.session_state.conversation.append(turn)
    with st.spinner("Nova is preparing your answer…"):
        _submit_corrected_question(api_key, turn, question)


def main() -> None:
    st.set_page_config(
        page_title="Nova • Voice Companion",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_dotenv()
    _init_state()
    if "voice_choice" not in st.session_state:
        st.session_state.voice_choice = "marin"

    with st.sidebar:
        st.markdown("## ✧ NOVA")
        st.caption("A little closer to the universe.")
        st.text_input(
            "OpenAI API key",
            type="password",
            key="api_key_input",
            placeholder="Paste your key here",
            help="Used only by this Streamlit session. You can also set OPENAI_API_KEY in .env.",
        )
        st.selectbox("Voice for new replies", VOICES, key="voice_choice")
        st.selectbox("Appearance", ("System", "Dark", "Light"), key="theme_mode")
        st.toggle("Animate the cosmos", key="motion")
        st.caption("New replies play aloud automatically. You can replay any reply below its message.")
        st.divider()
        st.caption("Your conversation stays in this browser session.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.processed_recordings = set()
            st.session_state.pending_audio = None
            st.session_state.autoplay_reply = None
            for name in ("audio_errors", "answer_errors", "transcription_errors"):
                st.session_state[name] = {}
            st.session_state.recorder_version += 1
            st.rerun()

    _apply_theme(st.session_state.theme_mode)
    api_key = st.session_state.api_key_input.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    with st.sidebar:
        st.caption("API key ready" if api_key else "Add a key to start")

    st.markdown(
        '<div class="app-bar">' + _nova_avatar() + '<div class="brand">'
        '<h1>NOVA</h1><p><span class="status-dot"></span>Online · your voice companion</p>'
        '</div><span class="sector">SECTOR 7G</span></div>', unsafe_allow_html=True,
    )
    with st.container(height=460, key="thread"):
        if not st.session_state.conversation:
            st.markdown(
                '<div class="welcome">' + _nova_avatar(True) +
                '<h2>Ready for <span class="text-cosmic">liftoff</span></h2>'
                '<p>Ask me anything about the universe —<br>or launch one of these missions.</p></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="missions"):
                columns = st.columns(2)
                for index, (icon, question) in enumerate((
                    ("◎", "What is a black hole?"), ("↗", "Tell me about Mars"),
                    ("✧", "How big is our galaxy?"), ("☀", "How do stars shine?"),
                )):
                    if columns[index % 2].button(f"{icon}   {question}", key=f"mission-{index}",
                                                use_container_width=True, disabled=not bool(api_key)):
                        _process_text(api_key, question)
                        st.rerun()

        conversation = st.session_state.conversation
        if conversation:

            for index, turn in enumerate(conversation, start=1):
                with st.chat_message("user"):
                    st.markdown(f'<div class="turn-label">You · {index:02d}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="transcript" dir="auto">{escape(turn["question"] or turn["original_transcript"] or "Transcribing your message…")}</div>', unsafe_allow_html=True
                    )
                    if turn["question"] and turn["question"] != turn["original_transcript"]:
                        st.caption(f"Original transcript: {turn['original_transcript']}")

                    with st.expander("Edit transcript"):
                        with st.form(f"edit-{turn['id']}"):
                            corrected = st.text_area(
                                "Correct the transcript",
                                value=turn["question"] or turn["original_transcript"],
                                key=f"transcript-{turn['id']}",
                            )
                            submitted = st.form_submit_button("Submit corrected question")
                        if submitted:
                            if not api_key:
                                st.error("Add your OpenAI API key in the sidebar before submitting.")
                            else:
                                _submit_corrected_question(api_key, turn, corrected)
                                st.rerun()

                    if turn["id"] in st.session_state.transcription_errors:
                        st.error(f"Transcription failed: {st.session_state.transcription_errors[turn['id']]}")
                        if st.button("Retry transcription", key=f"retry-transcript-{turn['id']}"):
                            if api_key:
                                try:
                                    turn["original_transcript"] = _transcribe(
                                        _client(api_key), turn["recording_bytes"]
                                    )
                                    turn["question"] = turn["original_transcript"]
                                    turn.pop("recording_bytes", None)
                                    st.session_state.transcription_errors.pop(turn["id"], None)
                                    st.session_state.pending_audio = None
                                    try:
                                        turn["answer"] = _answer(
                                            _client(api_key),
                                            _history_before(conversation, turn),
                                            turn["question"],
                                        )
                                        st.session_state.pending_audio = {
                                            "turn_id": turn["id"],
                                            "voice": st.session_state.voice_choice,
                                        }
                                    except (OpenAIError, ValueError) as exc:
                                        st.session_state.answer_errors[turn["id"]] = str(exc)
                                    st.rerun()
                                except (OpenAIError, ValueError) as exc:
                                    st.session_state.transcription_errors[turn["id"]] = str(exc)
                                    st.rerun()
                            else:
                                st.error("Add your OpenAI API key in the sidebar before retrying.")

                with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                    st.markdown('<div class="turn-label">Assistant</div>', unsafe_allow_html=True)
                    if turn["answer"]:
                        st.markdown(turn["answer"])
                    elif turn["id"] in st.session_state.answer_errors:
                        st.error(f"Answer generation failed: {st.session_state.answer_errors[turn['id']]}")
                        if st.button("Retry answer", key=f"retry-answer-{turn['id']}"):
                            if api_key and turn["question"]:
                                try:
                                    turn["answer"] = _answer(
                                        _client(api_key),
                                        _history_before(conversation, turn),
                                        turn["question"],
                                    )
                                    st.session_state.answer_errors.pop(turn["id"], None)
                                    st.session_state.pending_audio = {
                                        "turn_id": turn["id"],
                                        "voice": st.session_state.voice_choice,
                                    }
                                    st.rerun()
                                except (OpenAIError, ValueError) as exc:
                                    st.session_state.answer_errors[turn["id"]] = str(exc)
                                    st.rerun()
                            else:
                                st.error("Add your OpenAI API key before retrying.")
                    elif not turn["original_transcript"]:
                        st.caption("An answer will appear when transcription succeeds.")

                    if turn["answer"]:
                        selected_voice = st.selectbox(
                            "Playback voice", VOICES,
                            index=VOICES.index(turn.get("initial_voice", st.session_state.voice_choice)),
                            key=f"playback-{turn['id']}",
                            help="Saved voices replay without API calls. Generate a new voice with the button below.",
                        )
                        saved_audio = turn["audio"].get(selected_voice)
                        if saved_audio:
                            st.markdown(
                                f'<span class="ai-label">✦ AI-generated voice · {selected_voice}</span>',
                                unsafe_allow_html=True,
                            )
                            autoplay = st.session_state.autoplay_reply == (turn["id"], selected_voice)
                            st.audio(saved_audio, format="audio/wav", autoplay=autoplay)
                            if autoplay:
                                st.session_state.autoplay_reply = None
                        elif st.session_state.pending_audio and (
                            st.session_state.pending_audio["turn_id"] == turn["id"]
                            and st.session_state.pending_audio["voice"] == selected_voice
                        ):
                            st.caption(f"Preparing the {selected_voice} voice…")
                        else:
                            if selected_voice in st.session_state.audio_errors.get(turn["id"], {}):
                                st.error(
                                    f"Audio generation failed: "
                                    f"{st.session_state.audio_errors[turn['id']][selected_voice]}"
                                )
                                button_label = "Retry audio"
                            else:
                                button_label = f"Try another voice · {selected_voice}"
                            if st.button(button_label, key=f"audio-{turn['id']}-{selected_voice}"):
                                if not api_key:
                                    st.error("Add your OpenAI API key in the sidebar before generating audio.")
                                else:
                                    st.session_state.pending_audio = {
                                        "turn_id": turn["id"],
                                        "voice": selected_voice,
                                    }
                                    st.rerun()

    with st.container(key="composer"):
        with st.form("message-composer", clear_on_submit=True):
            text_column, send_column = st.columns([7, 1])
            question = text_column.text_input(
                "Message Nova", placeholder="Transmit a message to Nova…", label_visibility="collapsed",
                disabled=not bool(api_key),
            )
            send = send_column.form_submit_button("➤", disabled=not bool(api_key), use_container_width=True)
        if send and question.strip():
            _process_text(api_key, question.strip())
            st.rerun()
        st.caption("Speak to Nova · new replies play aloud automatically")

    
        recording = st.audio_input(
            "Record your question", key=f"voice_question-{st.session_state.recorder_version}",
            disabled=not bool(api_key), sample_rate=16000,
        )
        if recording is not None:
            audio_bytes = recording.getvalue()
            recording_id = hashlib.sha256(audio_bytes).hexdigest()
            if recording_id not in st.session_state.processed_recordings:
                if not api_key:
                    st.warning("Add your OpenAI API key in the sidebar to transcribe this recording.")
                else:
                    _process_recording(api_key, audio_bytes, recording_id)
                    st.session_state.recorder_version += 1
                    st.rerun()

    pending = st.session_state.pending_audio
    if pending and api_key:
        turn = next(
            (item for item in st.session_state.conversation if item["id"] == pending["turn_id"]),
            None,
        )
        st.session_state.pending_audio = None
        if turn and turn["answer"] and pending["voice"] not in turn["audio"]:
            with st.status(f"Generating {pending['voice']} audio…", expanded=True) as status:
                try:
                    audio_bytes = _generate_audio(api_key, turn["answer"], pending["voice"])
                    turn["audio"][pending["voice"]] = audio_bytes
                    st.session_state.autoplay_reply = (turn["id"], pending["voice"])
                    st.session_state.audio_errors.get(turn["id"], {}).pop(pending["voice"], None)
                    status.update(label="Voice reply ready.", state="complete", expanded=False)
                except AudioGenerationError as exc:
                    st.session_state.audio_errors.setdefault(turn["id"], {})[pending["voice"]] = str(exc)
                    status.update(label="Audio generation failed.", state="error", expanded=True)
            st.rerun()




if __name__ == "__main__":
    main()



