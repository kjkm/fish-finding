import pyroomacoustics as pra
from os.path import dirname, join as pjoin
from scipy.io import wavfile

CURRENT_DIR = 'room-acoustics'
AUDIO_DIR = 'audio'
FILE_NAME = '8,1640_StaatermanE-2018_Amphichthys-cryptocentrus_Boop-Grunt-Swoop.wav'

# ROOM SETUP
rt60 = 0.5  # seconds, reverb time
room_dim = [9, 7.5, 3.5]  # meters, room dimensions

e_absorption, max_order = pra.inverse_sabine(rt60, room_dim) # We invert Sabine's formula to obtain the parameters for the ISM simulator

room = pra.ShoeBox(
    room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order # Create the room
)

# LOAD AUDIO SOURCE
data_dir = pjoin(dirname(__file__), 'dataverse_files', 'Recordings').replace(CURRENT_DIR, AUDIO_DIR)
wav_fname = pjoin(data_dir, FILE_NAME)
samplerate, audio = wavfile.read(wav_fname)

# PLACE SOURCE IN ROOM
room.add_source([2.5, 3.73, 1.76], signal=audio, delay=1.3)