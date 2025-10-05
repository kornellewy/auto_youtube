from pathlib import Path
import soundfile as sf
import pyloudnorm as pyln

ROOT_DIR   = Path("/media/kornellewy/jan_dysk_3/auto_youtube/media/music")          # folder with mp3 files
LOUDNESS   = -14.0               # LUFS target (YouTube standard)

for mp3 in ROOT_DIR.glob("*.mp3"):
    try:
        data, rate = sf.read(mp3)          # load stereo or mono
        meter = pyln.Meter(rate)           # create BS.1770 meter
        loudness = meter.integrated_loudness(data)

        # loudness-normalise
        data_norm = pyln.normalize.loudness(data, loudness, LOUDNESS)

        # save back (SoundFile will auto-detect mp3 via extension)
        sf.write(mp3, data_norm, rate)
        print(f"Normalised {mp3.name}  {loudness:.1f} → {LOUDNESS} LUFS")
    except Exception as e:
        print(f"Skip {mp3.name}  ({e})")