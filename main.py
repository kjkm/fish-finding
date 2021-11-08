import matplotlib.pyplot as plt
import soundfile as sf

data, sample_rate = sf.read('audio/2021-10-23T19-25-59Z.wav', dtype='float32')

num_samples, num_channels = data.shape
print("samples ", num_samples, "channels ", num_channels)

colors = ["#008EFF", "#24FF00", "#B200FF", "#FF0000", "#FFC400", "#EF00FF"]

for channel in range(0, num_channels):
    plt.plot(data[:, channel], colors[channel])
    plt.xlabel("Time")
    plt.ylabel("Sound Pressure")
    plt.show()


