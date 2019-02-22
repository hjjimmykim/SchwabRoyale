import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.animation as manimation


ffmpegWriter = manimation.writers['ffmpeg']
metadata = dict(title = 'Test', artist='Tester', comment='Test')
writer = ffmpegWriter(fps=15, metadata=metadata)

N = 100 # Number of frames

fig = plt.figure()

with writer.saving(fig, "test.mp4", 100):
    for i in range(N):
        plt.clf() # Clear figure
        a = np.random.random([5,5])
        plt.imshow(a)
        writer.grab_frame()
