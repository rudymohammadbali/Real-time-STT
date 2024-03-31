import threading
import queue
import io
import logging

import pyaudio
import speech_recognition as sr

import time
from faster_whisper import WhisperModel


class STT:
    """Real-time Speech to Text class using Faster WhisperModel and speech_recognition."""

    def __init__(self, model_size: str = "medium.en", device: str = "cuda", compute_type: str = "float16",
                 language: str = "en", logging_level: str = None):
        """Initialize the STT object."""
        self.recorder = sr.Recognizer()
        self.data_queue = queue.Queue()
        self.transcription = ['']
        self.last_transcription = ""
        self.is_listening = True

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.default_mic = self.setup_mic()

        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        self.lock = threading.Lock()

        if logging_level:
            self.configure_logging(level=logging_level)

        self.thread = threading.Thread(target=self.transcribe)
        self.thread.setDaemon(True)
        self.thread.start()

        logging.info("Ready!\n")
        print("Ready!\n")

    def transcribe(self):
        """Transcribe the audio data from the queue."""
        while self.is_listening:
            audio_data = self.data_queue.get()

            if audio_data == 'STOP':
                break

            segments, info = self.model.transcribe(audio_data, beam_size=5, language=self.language, vad_filter=True)
            for segment in segments:
                text = segment.text.strip()
                logging.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, text))
                with self.lock:
                    self.transcription.append(text)
                    self.last_transcription = text

            self.data_queue.task_done()
            time.sleep(0.25)

    def recorder_callback(self, _, audio_data):
        """Callback function for the recorder."""
        audio = io.BytesIO(audio_data.get_wav_data())
        self.data_queue.put(audio)

    def listen(self):
        """Start listening to the microphone."""
        with sr.Microphone(device_index=self.default_mic) as source:
            self.recorder.adjust_for_ambient_noise(source)

        self.recorder.listen_in_background(source=source, callback=self.recorder_callback)

    def stop(self):
        """Stop the transcription process."""
        logging.info("Stopping...")
        logging.info(f"Transcription:\n {self.transcription}")
        self.is_listening = False
        self.data_queue.put("STOP")

    def get_last_transcription(self):
        """Get the last transcription and clear it."""
        with self.lock:
            text = self.last_transcription
            self.last_transcription = ""
        return text

    @staticmethod
    def setup_mic():
        """Set up the microphone."""
        p = pyaudio.PyAudio()
        default_device_index = None
        try:
            default_input = p.get_default_input_device_info()
            default_device_index = default_input["index"]
        except (IOError, OSError):
            logging.error("Default input device not found. Printing all input devices:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    logging.info(f"Device index: {i}, Device name: {info['name']}")
                    if default_device_index is None:
                        default_device_index = i

        if default_device_index is None:
            raise Exception("No input devices found.")

        return default_device_index

    @staticmethod
    def configure_logging(level: str = "INFO"):
        """
        Configure the logging level for the whole application.
        :param level: The desired logging level. Should be one of the following:
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        """
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logging.basicConfig(level=levels.get(level.upper(), logging.INFO))


# Usage
try:
    stt = STT()
    stt.listen()

    while stt.is_listening:
        last_transcription = stt.get_last_transcription()
        if len(last_transcription) > 0:
            print("You said: ", last_transcription)
            # If user said 'stop' then stop the transcription process by calling stt.stop()
            if "stop" in last_transcription.lower():
                stt.stop()

        time.sleep(1)

except KeyboardInterrupt:
    pass
