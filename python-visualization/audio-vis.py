import matplotlib.pyplot as plt
import soundfile as sf
from os import chdir
from os.path import dirname, join as pjoin

# FILE CONFIGURATIONS
FILE_NAME = '2021-10-23T19-25-59Z.wav'
AUDIO_DIR = 'audio'
# Changes directory from /python-visualization into /audio
chdir(pjoin(dirname(__file__), AUDIO_DIR))


data, sample_rate = sf.read(FILE_NAME, dtype='float32')

num_samples, num_channels = data.shape
print("samples ", num_samples, "channels ", num_channels)

colors = ["#008EFF", "#24FF00", "#B200FF", "#FF0000", "#FFC400", "#EF00FF"]

for channel in range(0, num_channels):
    plt.plot(data[:, channel], colors[channel])
    plt.xlabel("Time")
    plt.ylabel("Sound Pressure")
    plt.show()

plt.plot(data)
plt.xlabel("Time")
plt.ylabel("Sound Pressure")
plt.show()


