import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .planet_animation import PlanetAnimation


class RadialVelocityAnimation(PlanetAnimation):
    def __init__(self, star_path, planet_path, background_shape, star_scale, planet_scale,
                 inclination, radius, period,
                 r_star, rv_ax_position) -> None:
        super().__init__(star_path, planet_path, background_shape,
                         star_scale, planet_scale, inclination, radius, period)

        self.r_star = r_star
        self.setup_rv_ax(rv_ax_position)

    def setup_rv_ax(self, rv_ax_position):
        self.rv_ax: plt.Axes = self.ax.inset_axes(rv_ax_position)
        self.rv_ax.set_xticks([])
        self.rv_ax.set_yticks([])
        self.rv_ax.patch.set_alpha(0)
        self.rv_ax.spines['right'].set_visible(False)
        self.rv_ax.spines['top'].set_visible(False)
        self.rv_ax.set_xlabel('Time')
        self.rv_ax.set_ylabel('Radial Velocity')
        self.rv_ax.set_ylim(-1.2, 1.2)

    def initiate(self):
        super().initiate()
        self.ax.set_axis_off()
        self.rv_lc = LineCollection([], cmap='seismic', norm=plt.Normalize(-1, 1))
        self.rv_lc.set_array(np.array([]))
        self.rv_ax.add_collection(self.rv_lc)
        self.rv_last_point = None

    def draw_star_planet(self, planet_x, planet_y, planet_z):
        star_x = -planet_x/self.radius*self.r_star
        star_z = -planet_z/self.radius*self.r_star
        if planet_y > 0:
            img = self.draw_img_center(
                self.background, self.planet_img, center_x=int(planet_z), center_y=int(planet_x))
            img = self.draw_img_center(
                img, self.star_img, center_x=int(star_z), center_y=int(star_x))
        else:
            img = self.draw_img_center(
                self.background, self.star_img, center_x=int(star_z), center_y=int(star_x))
            img = self.draw_img_center(
                img, self.planet_img, center_x=int(planet_z), center_y=int(planet_x))
        return img

    def radial_velocity(self, time: float):
        phase = self.phase + 2*np.pi * time
        return -np.cos(phase)  # y velocity of the star, normalized

    def update_velocity(self, time):
        velocity = self.radial_velocity(time)
        point = [time, velocity]
        if self.rv_last_point is not None:
            self.rv_lc.set_segments(
                self.rv_lc.get_segments() + [np.array([self.rv_last_point, point])])
            self.rv_lc.set_array(np.append(self.rv_lc.get_array(), [velocity]))
        self.rv_last_point = point

    def animate(self, frame_i: int):
        time = frame_i / self.period
        x, y, z = self.planet_position_projected(time)

        img = self.draw_star_planet(x, y, z)
        self.ax.imshow(img, aspect='auto')
        self.update_velocity(time)

    def make_animation(self, nperiod):
        self.rv_ax.set_xlim(0, nperiod)
        return super().make_animation(nperiod)