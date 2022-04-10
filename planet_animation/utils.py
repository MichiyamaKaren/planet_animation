import numpy as np


def overlay_img_samesize(back: np.ndarray, front: np.ndarray)->np.ndarray:
    if front.shape[-1] == 3:
        return front

    front_alpha = (front[:, :, -1]).astype(float) / 255
    img = (np.einsum('ij,ijk->ijk', front_alpha, front.astype(float)) + \
        np.einsum('ij,ijk->ijk', 1-front_alpha, back.astype(float))).astype(front.dtype)
    return img


def overlay_img(back: np.ndarray, front: np.ndarray, pos_x: int, pos_y: int, exceed_handling='cut') ->np.ndarray:
    x_lim = pos_x + front.shape[0]
    y_lim = pos_y + front.shape[1]
    if x_lim > back.shape[0] or y_lim > back.shape[1]:
        if exceed_handling == 'pad':
            full_back = np.zeros(
                (max(x_lim, back.shape[0]), max(y_lim, back.shape[1]), 4),
                dtype=back.dtype)
            full_back[:back.shape[0], :back.shape[1]] = back
            back = full_back
        elif exceed_handling == 'cut':
            x_lim = min(x_lim, back.shape[0])
            y_lim = min(y_lim, back.shape[1])
            front = front[:x_lim-pos_x, :y_lim-pos_y]

    overlayed = back.copy()
    overlayed[pos_x:x_lim, pos_y:y_lim] = overlay_img_samesize(
        back[pos_x:x_lim, pos_y:y_lim], front)
    return overlayed


def alpha_weighted_covered_pixels(back: np.ndarray, front: np.ndarray, pos_x: int, pos_y: int):
    x_max = min(pos_x+front.shape[0], back.shape[0])
    y_max = min(pos_y+front.shape[1], back.shape[1])
    x_min = max(0, pos_x)
    y_min = max(0, pos_y)

    if x_max <= x_min or y_max <= y_min:
        return 0

    if back.shape[-1] == 3:
        back_weight = np.ones((x_max-x_min, y_max-y_min), dtype=float)
    else:
        back_weight = back[x_min:x_max, y_min:y_max, -1].astype(float)/255
    if front.shape[-1] == 3:
        front_weight = np.ones((x_max-x_min, y_max-y_min), dtype=float)
    else:
        front_weight = front[:x_max-x_min, :y_max-y_min, -1].astype(float)/255
    cover = front_weight*back_weight
    return np.sum(cover)
