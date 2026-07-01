import imageio.v3 as iio
from IPython.display import HTML
import base64
import matplotlib.pyplot as plt
import io
import uuid
import numpy as np

import io, base64, uuid
import numpy as np
import imageio.v3 as iio
from IPython.display import HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def show_video_jshtml(video, fps=20):
    """
    """
    fig, ax = plt.subplots()
    im = ax.imshow(video[0])
    ax.axis('off')

    def update(i):
        im.set_data(video[i])
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(video), interval=1000/fps, blit=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())

def show_depth_video_jshtml(depth_video, fps=20, cmap='Spectral'):
    """
    """
    vmin, vmax = depth_video.min(), depth_video.max()

    fig, ax = plt.subplots()
    im = ax.imshow(depth_video[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(i):
        im.set_data(depth_video[i])
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(depth_video), interval=1000/fps, blit=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())
