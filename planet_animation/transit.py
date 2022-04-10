import numpy as np

import matplotlib.pyplot as plt

from .planet_animation import PlanetAnimation
from .utils import alpha_weighted_covered_pixels


class TransitAnimation(PlanetAnimation):
    def __init__(self, star_path, planet_path, background_shape, star_scale, planet_scale,
                 inclination, radius, period,
                 lum_ax_position) -> None:
        super().__init__(star_path, planet_path, background_shape,
                         star_scale, planet_scale, inclination, radius, period)

        self.star_lum = np.sum(self.star_img[:, :, -1]) / 255
        self.setup_lum_ax(lum_ax_position)

    def setup_lum_ax(self, lum_ax_position):
        self.lum_ax: plt.Axes = self.ax.inset_axes(lum_ax_position)
        self.lum_ax.set_xticks([])
        self.lum_ax.set_yticks([])
        self.lum_ax.patch.set_alpha(0)
        self.lum_ax.spines['right'].set_visible(False)
        self.lum_ax.spines['top'].set_visible(False)
        self.lum_ax.set_xlabel('Time')
        self.lum_ax.set_ylabel('Luminosity')

        planet_lum = np.sum(self.planet_img[:,:,-1])/255
        lum_depth = planet_lum/self.star_lum
        self.lum_ax.set_ylim(1-lum_depth*6/5, 1+lum_depth/5)

    def initiate(self):
        super().initiate()
        self.ax.set_axis_off()
        self.lum_line, = self.lum_ax.plot([], [])

    def transited_relative_luminosity(self, planet_x, planet_y, planet_z):
        if planet_y > 0:
            return 1
        planet_pos_x = int(planet_z) - self.planet_img.shape[0]//2 + self.star_img.shape[0]//2
        planet_pos_y = int(planet_x) - self.planet_img.shape[1]//2 + self.star_img.shape[1]//2
        return 1 - alpha_weighted_covered_pixels(
            self.star_img, self.planet_img, planet_pos_x, planet_pos_y) / self.star_lum

    def update_luminosity(self, planet_x, planet_y, planet_z, time):
        luminosity = self.transited_relative_luminosity(planet_x, planet_y, planet_z)
        t, lum = self.lum_line.get_data()
        self.lum_line.set_data(np.hstack((t, [time])), np.hstack((lum, [luminosity])))

    def animate(self, frame_i: int):
        time = frame_i / self.period
        x, y, z = self.planet_position_projected(time)

        img = self.draw_star_planet(x, y, z)
        self.ax.imshow(img, aspect='auto')
        self.update_luminosity(x, y, z, time)
    
    def make_animation(self, nperiod):
        self.lum_ax.set_xlim(0, nperiod)
        return super().make_animation(nperiod)
