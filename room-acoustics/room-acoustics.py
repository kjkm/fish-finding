import pyroomacoustics as pra
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import time
import samplerate

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
ROOM_WIDTH = 100.0
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


# Streamlines the whole directory management kerfuffle
def get_path(filename: str) -> str:
    data_dir = pjoin(dirname(__file__), 'dataverse_files', 'Recordings').replace(CURRENT_DIR, AUDIO_DIR)
    # TODO: Find a way to make it so that 'dataverse-files' and 'Recordings' can be moved up top to be config variables
    return pjoin(data_dir, filename)


# Generates a wav file from a given room configuration at a given location
def generate_wav(file: str, output_path: str, reverb: float, dims: tuple[float, float, float], source: tuple[float, float, float], delay: float = 1.3):
    samplerate, audio = wavfile.read(get_path(file))

    plot(audio)  # Plot source signal before transformation

    # Configure room
    rt60 = reverb  # seconds, reverb time
    room_dim = [dims[0], dims[1], dims[2]]  # meters, room dimensions
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=samplerate,
        materials=pra.Material(e_absorption),
        max_order=10  # hardcoded a lower reflection order
    )

    # Configure sensor array
    square_array = [center_point_on_origin(ORIGIN, sensor) for sensor in SQUARE_DELTA]
    sensor_locations = np.transpose(square_array)

    # Add sensor array to room
    room.add_microphone_array(sensor_locations)

    # Generate WAV file
    room.fs = samplerate
    room.add_source([source[0], source[1], source[2]], signal=audio, delay=delay)
    room.compute_rir()

    plot(room.rir[1][0])  # Plot RIR between Mic 1 and Src 0

    room.simulate()
    room.mic_array.to_wav(
        output_path,
        norm=True,
        bitdepth=np.float32,
    )


# TODO: After configuring room, call generate_wav for every sound in the data set at several random different locs.
def main():
    # Generate WAV file
    source_loc = (2.5, 3.73, 1.76)

    generate_wav(
        FILE_NAME,
        f"output/generated{time.time()}.wav",
        1.0,
        (ROOM_WIDTH, ROOM_HEIGHT, ROOM_WIDTH),
        source_loc
    )


if __name__ == "__main__":
    main()
