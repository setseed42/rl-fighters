import numpy as np
import os
import pyglet
from pyglet.window import key
import pkg_resources

class GameEnv(object):
    def __init__(self, num_chars: int):
        self.num_chars: int = num_chars
        self.window_width: int = 300
        self.window_height: int = 400
        self.state = None
        self.image_map = self._get_art()
        self.char_width = self.image_map['standing_up'].width
        self.char_height = self.image_map['standing_up'].height
        self.action_matrix = [
            [key.RIGHT, 'right', np.array([10, 0]), [1, 0, 0, 0]],
            [key.LEFT, 'left', np.array([-10, 0]), [0, 1, 0, 0]],
            [key.UP, 'up', np.array([0, 10]), [0, 0, 1, 0]],
            [key.DOWN, 'down', np.array([0, -10]), [0, 0, 0, 1]],
        ]
        self.anim_map = {l[0]: l[1] for l in self.action_matrix}
        self.key_map = {l[0]: l[2] for l in self.action_matrix}
        self.direct_map = {tuple(l[2]): l[1] for l in self.action_matrix}
        self.reset_state()

    def reset_state(self):
        height_splits = np.linspace(
            self.char_height, self.window_width-self.char_height, num=self.num_chars+1)
        y_coords = np.array([
            np.random.randint(height_splits[i]+10, height_splits[i+1]-10)
            for i in range(self.num_chars)
        ]).reshape(self.num_chars, 1)
        np.random.shuffle(y_coords)
        self.state = {
            'stance': np.resize('standing_right', self.num_chars).astype('<U16'),
            'coords': np.hstack([
                np.random.randint(
                    self.char_width, self.window_width-self.char_width, size=(self.num_chars, 1)),
                y_coords
            ])
        }
        return self

    def _get_art(self):
        DATA_PATH = pkg_resources.resource_filename(
            'gym_fighters', 'envs/assets')
        print(DATA_PATH)
        imgs = os.listdir(DATA_PATH)
        loader = pyglet.resource.Loader([DATA_PATH])

        def get_animation_name(img):
            return img \
                .replace('knight_', '') \
                .replace('.png', '')

        def get_image_resource(img):
            path = f'{DATA_PATH}/{img}'
            #char_resource = loader.image(path)
            char_resource = pyglet.image.load(path)
            char_resource.anchor_x = char_resource.width // 2
            char_resource.anchor_y = char_resource.height // 2
            return char_resource

        return {
            get_animation_name(img): get_image_resource(img)
            for img in imgs
            if '2' not in img
            and '.png' in img
        }

    def set_state(self, state):
        self.state = state
        return self
