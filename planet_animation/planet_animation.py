import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as mpani

from .utils import overlay_img


class PlanetAnimation:
    def __init__(self, star_path, planet_path,
                 background_shape, star_scale, planet_scale,
                 inclination, radius, period) -> None:
        dpi = plt.rcParams['figure.dpi']
        self.fig = plt.figure(figsize=(background_shape[1]/dpi, background_shape[0]/dpi))
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.fig.add_axes(self.ax)

        self.star_img = cv2.cvtColor(cv2.imread(star_path, -1), cv2.COLOR_BGRA2RGBA)
        self.planet_img = cv2.cvtColor(cv2.imread(planet_path, -1), cv2.COLOR_BGRA2RGBA)
        self.star_img: np.ndarray = cv2.resize(
            self.star_img, (int(self.star_img.shape[1]*star_scale), int(self.star_img.shape[0]*star_scale)))
        self.planet_img: np.ndarray = cv2.resize(
            self.planet_img, (int(self.planet_img.shape[1]*planet_scale), int(self.planet_img.shape[0]*planet_scale)))

        self.background: np.ndarray = 255*np.ones(list(background_shape)+[4], dtype=self.star_img.dtype)
        self.background_with_star = self.draw_img_center(
            self.background, self.star_img, center_x=0, center_y=0)

        self.inclination = np.deg2rad(inclination)
        self.period = period
        self.radius = radius
        self.phase = 0

    def initiate(self):
        self.fig.subplots_adjust(
            left=None, bottom=None, right=None, wspace=None, hspace=None)

    def planet_position_projected(self, time: float):
        phase = self.phase + 2*np.pi*time
        x = self.radius*np.cos(phase)
        y = self.radius*np.sin(phase)*np.cos(self.inclination)
        z = self.radius*np.sin(phase)*np.sin(self.inclination)
        return x, y, z

    def draw_img_center(self, back: np.ndarray, front: np.ndarray, center_x: int, center_y: int):
        origin_x = self.background.shape[0]//2
        origin_y = self.background.shape[1]//2

        return overlay_img(
            back, front,
            pos_x=origin_x+center_x-front.shape[0]//2,
            pos_y=origin_y+center_y-front.shape[1]//2)

    def draw_star_planet(self, planet_x, planet_y, planet_z):
        if planet_y > 0:
            img = self.draw_img_center(
                self.background, self.planet_img, center_x=int(planet_z), center_y=int(planet_x))
            img = self.draw_img_center(
                img, self.star_img, center_x=0, center_y=0)
        else:
            img = self.draw_img_center(
                self.background_with_star, self.planet_img, center_x=int(planet_z), center_y=int(planet_x))
        return img

    def animate(self, frame_i: int):
        time = frame_i / self.period
        x, y, z = self.planet_position_projected(time)
        img = self.draw_star_planet(x, y, z)

        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.imshow(img, aspect='auto')

    def make_animation(self, nperiod):
        return mpani.FuncAnimation(
            fig=self.fig, func=self.animate,
            frames=int(nperiod*self.period), init_func=self.initiate)
