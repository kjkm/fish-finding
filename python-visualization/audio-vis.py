import matplotlib.pyplot as plt
import soundfile as sf
import os

FILE_NAME = '2021-10-23T19-25-59Z.wav'
THIS_DIR = 'python-visualization'
AUDIO_DIR = 'audio'

path = os.path.dirname(__file__)
path = path.replace(THIS_DIR, AUDIO_DIR)
os.chdir(path)
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


