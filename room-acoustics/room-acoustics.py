import pyroomacoustics
import pyroomacoustics as pra
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import time

# FILE CONFIGURATIONS
CURRENT_DIR = 'room-acoustics'
AUDIO_DIR = 'audio'
AUDIO_SUB_DIRS = []
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


# Given an origin point expressed in global coordinates and a point expressed in local coordinates, convert the point to
# global coordinates, relative to the origin. The origin and point should exist within the same coordinate system ie
# be lists of equal length.
def center_point_on_origin(origin: list[int], point: list[int]) -> list[int]:
    return [a+b for a, b in zip(origin, point)]


# So I don't need to dedicated two whole lines to plotting some data
def plot(data: list[np.ndarray]):
    plt.plot([row for row in data])
    plt.show()


# Accepts a room, a source file, an output path, a location, and an optional delay and saves a reverberant
# transformation of the source signal to a wav file in the specified location.
def generate_wav(room: pra.Room, filename: str, output_path: str, source_location: list[int], source_delay: float = 1.3):
    sample_rate, data = wavfile.read(filename)
    plot(data) # Plot source signal before transformation
    room.add_source(source_location, signal=data, delay=source_delay)
    room.compute_rir()
    plot(room.rir[1][0]) # Plot RIR between Mic 1 and Src 0
    room.simulate()
    # TODO: Plot source signal after it gets convolved with the RIR
    room.mic_array.to_wav(
        output_path,
        norm=True,
        bitdepth=np.float32,
    )


def get_path(filename: str) -> str:
    data_dir = pjoin(dirname(__file__), 'dataverse_files', 'Recordings').replace(CURRENT_DIR, AUDIO_DIR)
    return pjoin(data_dir, filename)


def main():
    samplerate, audio = wavfile.read(get_path(FILE_NAME))
    # TODO: input file is currently read in here, because we need the samplerate, and also in the generate_wav function

    # Configure room
    rt60 = 1.0  # seconds, reverb time
    room_dim = [ROOM_WIDTH, ROOM_HEIGHT, ROOM_WIDTH]  # meters, room dimensions
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=samplerate, materials=pra.Material(e_absorption), max_order=10 # hardcoded a lower reflection order
    )

    # Configure sensor array
    square_array = [center_point_on_origin(ORIGIN, sensor) for sensor in SQUARE_DELTA]
    sensor_locations = np.transpose(square_array)

    # Add sensor array to room
    room.add_microphone_array(sensor_locations)

    #Generate WAV file
    location = [2.5, 3.73, 1.76]
    generate_wav(room, get_path(FILE_NAME), f"output/generated{time.time()}.wav", location)


if __name__ == "__main__":
    main()