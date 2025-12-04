from gtts import gTTS
from playsound3 import playsound
from tempfile import NamedTemporaryFile
import os


def speak(text: str, lang: str = "en", slow: bool = False) -> None:
    """
    Convert text to speech using gTTS and play it immediately.

    :param text: Text to speak.
    :param lang: Language code, e.g. 'en', 'de', 'fr'.
    :param slow: If True, speak slower.
    """
    if not text or not text.strip():
        return

    # Create temp MP3 file
    with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tmp_path = fp.name
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.write_to_fp(fp)

    try:
        # playsound3 blocks until finished by default (block=True)
        playsound(tmp_path, block=True)
    finally:
        # Clean up file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

if __name__ == "__main__":
    speak("Bonjour Tristesse", lang="fr")