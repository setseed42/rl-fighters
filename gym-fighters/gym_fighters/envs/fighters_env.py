import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_fighters.envs.character import Character
from gym_fighters.envs.game_env import GameEnv

class FightersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_chars=2, n_rays=4):
        self.num_chars = num_chars
        self.n_rays = n_rays
        self.game_env = GameEnv(num_chars)
        self.chars = [
            Character(
                char_index=i,
                game_env=self.game_env,
                other_chars=self._get_other_chars(i),
                n_rays=n_rays,
            )
            for i in range(num_chars)
        ]
        self.vision = np.array([char.get_vision() for char in self.chars])
        self.action_choices = [
            key.RIGHT,
            key.LEFT,
            key.UP,
            key.DOWN,
            key.SPACE
        ]
        self.observations_dim = n_rays * 2
        self.window=None

    def _make_window(self):
        self.window = pyglet.window.Window(
            width=self.game_env.window_width,
            height=self.game_env.window_height,
        )

    def step(self, action, char_ix):
        new_state = self.chars[char_ix].update_char_position(action, self.game_env.state)
        self.game_env = self.game_env.set_state(new_state)
        if sum(self.game_env.state['stance'] == 'dead') == (self.num_chars - 1):
            done = True
            reward = (self.game_env.state['stance']!='dead').astype(float)
            reward = (reward * 2) - 1
            return None, reward, done, None
        else:
            reward = np.zeros(self.num_chars)
            done = False
            self.vision = np.array([
                char.get_vision()
                for char in self.chars
            ])
            return self.vision, reward, done, None

    def reset(self):
        if self.window is not None:
            self.window.close()
            self.window=None
        self.game_env = self.game_env.reset_state()
        self.chars = [
            Character(
                char_index=i,
                game_env=self.game_env,
                other_chars=self._get_other_chars(i),
                n_rays=self.n_rays,
            )
            for i in range(self.num_chars)
        ]
        return self.vision

    def render(self, mode='human', close=False):
        if self.window is None:
            self._make_window()
        def create_batch():
            batch = pyglet.graphics.Batch()
            redundant_but_necessary_for_batch_to_work = []
            return batch, redundant_but_necessary_for_batch_to_work

        pyglet.clock.tick()

        @self.window.event
        def on_draw():
            self.window.clear()
            batch = create_batch()
            self._sprite(
                batch,
                image_name=self.game_env.state['stance'],
                coords=self.game_env.state['coords'],
            )

            batch[0].draw()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()

    def _len_of_one(self, *objs):
        for obj in objs:
            if hasattr(obj, '__len__'):
                return len(obj)
        return 1

    def _sprite(self, batch, image_name, coords):
        length = self._len_of_one(image_name, coords)
        image_name = np.resize(image_name, length)
        x = np.resize(coords[:,0], length)
        y = np.resize(coords[:,1], length)

        for _image_name, _x, _y in zip(image_name, x, y):
            img = self.game_env.image_map[_image_name]
            sprite = pyglet.sprite.Sprite(img, x=_x, y=_y, batch=batch[0])
            batch[1].append(sprite)


    def _get_other_chars(self, index: int) -> np.ndarray:
        other_chars = np.arange(self.num_chars)
        return other_chars[other_chars != index]





if __name__ == "__main__":
    import time
    n_chars=2
    env = FightersEnv(2, 4)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = state
        step = 0
        done = False
        while not done:
            step += 1
            env.render()
            for char_ix in range(n_chars):
                action = np.random.choice(env.action_choices, 1)[0]
                next_state, reward, done, info = env.step(action, char_ix)
                if done:
                    print(f'game {run} done')
                    break
