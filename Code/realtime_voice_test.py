#!/usr/bin/env python3
"""
OpenAI Realtime-2 voice test for the KI-Telefon project.

Place this file in the same directory as:
    config.py
    openai_ws.py
    roles.py

Default:
    Tests all ten currently documented Realtime voices once.

Optional:
    --roles
        Tests every entry from roles.py using its configured voice and speed.
        Voices that are assigned to several roles are therefore played several times.
"""

import argparse
import base64
import json
import ssl
import time
import wave
from collections import defaultdict
from pathlib import Path

import pyaudio

from openai_ws import (
    API_KEY,
    REALTIME_MODEL,
    WS_URL,
    create_connection_with_ipv4,
    send_json,
)
from roles import role as ROLE_LIST


ALL_REALTIME_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)

TEST_TEXT = (
    "Guten Tag. Dies ist ein kurzer deutscher Sprachtest mit ruhiger Betonung, "
    "natürlichem Sprechtempo und klarer Aussprache."
)

RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # PCM16
PAUSE_BETWEEN_SAMPLES_S = 0.8
OUTPUT_DIRECTORY = Path("voice_samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testet die Stimmen des OpenAI-Realtime-2-Modells."
    )
    parser.add_argument(
        "--roles",
        action="store_true",
        help=(
            "Alle Einträge aus roles.py mit voice_id und speed testen. "
            "Ohne diese Option werden alle zehn Realtime-Stimmen einmal getestet."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Keine WAV-Dateien speichern.",
    )
    return parser.parse_args()


def wait_for_event(ws, expected_type):
    """Wait until the requested Realtime event arrives."""
    while True:
        raw_message = ws.recv()
        if not raw_message:
            raise RuntimeError("OpenAI hat die WebSocket-Verbindung beendet.")

        event = json.loads(raw_message)
        event_type = event.get("type", "")

        if event_type == "error":
            raise RuntimeError(
                "OpenAI Realtime API: "
                + json.dumps(event, ensure_ascii=False)
            )

        if event_type == expected_type:
            return event


def create_session(ws, voice_id, speed):
    """
    Configure one fresh Realtime session.

    A separate WebSocket session is used for every sample because the voice
    cannot be changed after audio has already been emitted in that session.
    """
    wait_for_event(ws, "session.created")

    send_json(
        ws,
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": REALTIME_MODEL,
                "output_modalities": ["audio"],
                "instructions": (
                    "Du führst ausschließlich einen Sprachtest aus. "
                    "Sprich den vom Benutzer gesendeten Text exakt und vollständig aus. "
                    "Füge nichts hinzu, lasse nichts weg und beantworte den Inhalt nicht. "
                    "Sprich auf Deutsch, natürlich, ruhig und deutlich."
                ),
                "audio": {
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": RATE,
                        },
                        "voice": voice_id,
                        "speed": speed,
                    },
                },
                "reasoning": {
                    "effort": "low",
                },
            },
        },
        "session.update",
    )

    wait_for_event(ws, "session.updated")


def request_audio(ws, text):
    """Send the test sentence and collect the returned PCM16 audio."""
    send_json(
        ws,
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            },
        },
        "voice_test_text",
    )

    send_json(
        ws,
        {
            "type": "response.create",
        },
        "voice_test_response",
    )

    audio = bytearray()
    transcript = ""

    while True:
        raw_message = ws.recv()
        if not raw_message:
            raise RuntimeError("OpenAI hat die WebSocket-Verbindung beendet.")

        event = json.loads(raw_message)
        event_type = event.get("type", "")

        if event_type == "error":
            raise RuntimeError(
                "OpenAI Realtime API: "
                + json.dumps(event, ensure_ascii=False)
            )

        if event_type in ("response.audio.delta", "response.output_audio.delta"):
            delta = event.get("delta")
            if delta:
                audio.extend(base64.b64decode(delta))

        elif event_type in (
            "response.audio_transcript.done",
            "response.output_audio_transcript.done",
        ):
            transcript = event.get("transcript", "")

        elif event_type == "response.done":
            response = event.get("response", {})
            status = response.get("status", "unknown")

            if status != "completed":
                details = response.get("status_details")
                raise RuntimeError(
                    f"Response endete mit Status {status}: {details}"
                )
            break

    if not audio:
        raise RuntimeError("Es wurden keine Audiodaten empfangen.")

    return bytes(audio), transcript


def generate_sample(voice_id, speed):
    """Open one Realtime-2 session and generate one sample."""
    ws = None

    try:
        ws = create_connection_with_ipv4(
            WS_URL,
            header=[
                f"Authorization: Bearer {API_KEY}",
                "OpenAI-Safety-Identifier: ki-telefon-local",
            ],
            sslopt={"cert_reqs": ssl.CERT_REQUIRED},
            timeout=10,
        )
        ws.settimeout(30.0)

        create_session(ws, voice_id, speed)
        return request_audio(ws, TEST_TEXT)

    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def play_pcm(pcm_data):
    """Play raw PCM16 audio through the default output device."""
    audio = pyaudio.PyAudio()
    stream = None

    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            output=True,
        )
        stream.write(pcm_data)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


def save_wav(filename, pcm_data):
    """Save PCM16 data as a standard WAV file."""
    filename.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(RATE)
        wav_file.writeframes(pcm_data)


def build_all_voice_samples():
    """
    Build one comparable test for every documented Realtime voice.

    roles.py is used to show which KI-Telefon roles currently use a voice.
    Unassigned voices are still included so all ten voices are tested.
    """
    roles_by_voice = defaultdict(list)

    for configured_role in ROLE_LIST:
        roles_by_voice[configured_role["voice_id"]].append(
            configured_role["name"]
        )

    samples = []

    for voice_id in ALL_REALTIME_VOICES:
        assigned_roles = roles_by_voice.get(voice_id, [])

        samples.append(
            {
                "label": voice_id,
                "voice_id": voice_id,
                "speed": 1.0,
                "roles": assigned_roles,
                "filename": f"{voice_id}.wav",
            }
        )

    return samples


def build_role_samples():
    """Build one sample for every actual role configuration."""
    samples = []

    for number, configured_role in enumerate(ROLE_LIST, start=1):
        safe_name = (
            configured_role["name"]
            .lower()
            .replace(" ", "_")
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )

        samples.append(
            {
                "label": configured_role["name"],
                "voice_id": configured_role["voice_id"],
                "speed": configured_role.get("speed", 1.0),
                "roles": [configured_role["name"]],
                "filename": (
                    f"{number:02d}_{safe_name}_"
                    f"{configured_role['voice_id']}.wav"
                ),
            }
        )

    return samples


def main():
    args = parse_args()

    if not API_KEY or API_KEY == "HIER OPENAI API KEY EINTRAGEN":
        raise SystemExit(
            "FEHLER: OPENAI_API_KEY ist in config.py nicht gesetzt."
        )

    samples = (
        build_role_samples()
        if args.roles
        else build_all_voice_samples()
    )

    mode = "Rollen aus roles.py" if args.roles else "alle Realtime-Stimmen"

    print(f"Modell: {REALTIME_MODEL}")
    print(f"Modus: {mode}")
    print(f"Testsatz: {TEST_TEXT}")
    print(f"Anzahl Tests: {len(samples)}")

    for index, sample in enumerate(samples, start=1):
        print()
        print("=" * 72)
        print(f"[{index}/{len(samples)}] {sample['label']}")
        print(f"Voice: {sample['voice_id']}")
        print(f"Speed: {sample['speed']}")

        if sample["roles"]:
            print("Verwendet von: " + ", ".join(sample["roles"]))
        else:
            print("Verwendet von: derzeit keiner Rolle in roles.py")

        try:
            pcm_data, transcript = generate_sample(
                sample["voice_id"],
                sample["speed"],
            )

            if transcript:
                print(f"KI-Transkript: {transcript}")

            if not args.no_save:
                output_file = OUTPUT_DIRECTORY / sample["filename"]
                save_wav(output_file, pcm_data)
                print(f"Gespeichert: {output_file}")

            print("Wiedergabe ...")
            play_pcm(pcm_data)

        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            return
        except Exception as exc:
            print(f"FEHLER: {exc}")

        time.sleep(PAUSE_BETWEEN_SAMPLES_S)

    print()
    print("Sprachtest abgeschlossen.")


if __name__ == "__main__":
    main()
