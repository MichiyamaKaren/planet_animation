# %%
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from planet_animation import TransitAnimation

# %%
pa = TransitAnimation(
    star_path='lanzhu.png', planet_path='earth.png',
    background_shape=(540, 960), star_scale=1/2, planet_scale=1/30,
    inclination=-10, radius=300, period=50,
    lum_ax_position=[5/96, 4/54, 20/96*1.5, 15/54])
pa.phase = np.deg2rad(150)

animation = pa.make_animation(nperiod=2)
start = datetime.now()
animation.save('transit.gif', fps=10, writer='imagemagick')
end = datetime.now()
print('used time:', end-start)