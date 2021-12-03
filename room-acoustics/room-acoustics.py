import pyroomacoustics as pra
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import time

# FILE CONFIGURATIONS
CURRENT_DIR = 'room-acoustics'
AUDIO_DIR = 'audio'
FILE_NAME = '8,1640_StaatermanE-2018_Amphichthys-cryptocentrus_Boop-Grunt-Swoop.wav'

# ROOM CONFIGURATIONS
SENSOR_DELTA = 0.317873
SQUARE_DELTA = [
    [0, 0, -SENSOR_DELTA],
    [0, 0, SENSOR_DELTA],
    [-SENSOR_DELTA, 0, 0],
    [SENSOR_DELTA, 0, 0]
]
ROOM_HEIGHT = 11.35
ROOM_WIDTH = 100
ORIGIN = [ROOM_WIDTH/2, 0, ROOM_WIDTH/2]


# Used to convert relative position to global position
def center_on_origin(origin, point):
    return [a+b for a, b in zip(origin, point)]


# LOAD AUDIO SOURCE
data_dir = pjoin(dirname(__file__), 'dataverse_files', 'Recordings').replace(CURRENT_DIR, AUDIO_DIR)
wav_fname = pjoin(data_dir, FILE_NAME)
samplerate, audio = wavfile.read(wav_fname)

plt.plot(audio) # Audio before simulation
plt.show()

# ROOM SETUP
rt60 = 1.0  # seconds, reverb time
room_dim = [ROOM_WIDTH, ROOM_HEIGHT, ROOM_WIDTH]  # meters, room dimensions

e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

print(max_order)
room = pra.ShoeBox(
    room_dim, fs=samplerate, materials=pra.Material(e_absorption), max_order=10 # hardcoded a lower reflection order
)

# PLACE SOURCE IN ROOM
room.add_source([2.5, 3.73, 1.76], signal=audio, delay=1.3)

# CONFIGURE SENSOR ARRAY
square_array = [center_on_origin(ORIGIN, sensor) for sensor in SQUARE_DELTA]

mic_locs = np.c_[
    square_array[0],  # mic 1
    square_array[1],  # mic 2
    square_array[2],  # mic 3
    square_array[3],  # mic 4
]

# PLACE ARRAY IN ROOM
room.add_microphone_array(mic_locs)

# COMPUTE RIR
room.compute_rir()

# PLOT RIR BETWEEN MIC 1 AND SOURCE 0
plt.plot(room.rir[1][0])
plt.show()

# CONVOLVE RIR AND SOURCES
room.simulate()

# PLOT SIGNALS
plt.plot(room.mic_array.signals[1, :])
plt.plot(room.mic_array.signals[0, :])
plt.plot(room.mic_array.signals[2, :])
plt.plot(room.mic_array.signals[3, :])
plt.show()

# WRITE TO WAV FILE
room.mic_array.to_wav(
    f"output/fishsound_{time.time()}.wav",
    norm=True,
    bitdepth=np.float32
)


