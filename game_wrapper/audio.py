import threading
import time

import numpy as np
import pygame


class AudioSystem:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        try:
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
            self.enabled = True
        except Exception as e:
            print(f"Audio init failed: {e}")
            self.enabled = False

    def play_tone(self, frequency, duration, volume=0.5, type="sine"):
        if not self.enabled:
            return

        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)

        if type == "sine":
            wave = np.sin(2 * np.pi * frequency * t)
        elif type == "square":
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        elif type == "saw":
            wave = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif type == "noise":
            wave = np.random.uniform(-1, 1, n_samples)

        # Envelope (Attack/Decay)
        attack = int(n_samples * 0.1)
        decay = int(n_samples * 0.2)
        env = np.ones(n_samples)
        env[:attack] = np.linspace(0, 1, attack)
        env[-decay:] = np.linspace(1, 0, decay)

        wave = wave * env * volume

        # Stereo (duplicate channel)
        stereo_wave = np.column_stack((wave, wave))

        # Convert to 16-bit PCM
        sound_data = (stereo_wave * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_data)
        sound.play()

    def play_scan_sound(self):
        # Sci-fi sweep
        threading.Thread(target=self._scan_sweep).start()

    def _scan_sweep(self):
        if not self.enabled:
            return
        # Create a frequency sweep
        duration = 0.5
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        freqs = np.linspace(400, 1200, n_samples)
        wave = np.sin(2 * np.pi * freqs * t) * 0.3

        sound_data = (np.column_stack((wave, wave)) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_data)
        sound.play()

    def play_engine_hum(self, thrust_level):
        # Placeholder
        pass


class MusicGenerator:
    def __init__(self, audio_sys):
        self.audio = audio_sys
        self.running = False
        self.thread = None
        self.base_freq = 110  # A2
        self.scale = [0, 2, 4, 7, 9]  # Pentatonic

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _loop(self):
        while self.running and self.audio.enabled:
            # Procedural ambient
            if np.random.random() < 0.3:
                # Play a note
                note_idx = np.random.choice(self.scale)
                octave = np.random.choice([1, 2, 4])
                freq = self.base_freq * octave * (2 ** (note_idx / 12.0))

                duration = np.random.uniform(2.0, 4.0)
                vol = np.random.uniform(0.1, 0.2)

                # We need a non-blocking play or just fire and forget
                # audio.play_tone blocks? No, it uses make_sound.play() which is async mixer.
                # But creating the array takes time.
                self.audio.play_tone(freq, duration, vol, "sine")

            time.sleep(np.random.uniform(0.5, 2.0))


audio = AudioSystem()
music = MusicGenerator(audio)
