# %%
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from planet_animation import PlanetAnimation

# %%
pa = PlanetAnimation(
    star_path='lanzhu.png', planet_path='earth.png',
    background_shape=(540, 960), star_scale=1/2, planet_scale=1/30,
    inclination=-10, radius=300, period=50)

animation = pa.make_animation(nperiod=1)
start = datetime.now()
animation.save('planet.gif', fps=10, writer='imagemagick')
end = datetime.now()
print('used time:', end-start)

# GIF to video:
# ffmpeg -f gif -i planet.gif planet.mp4