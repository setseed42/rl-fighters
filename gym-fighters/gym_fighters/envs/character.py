from gym_fighters.envs.game_env import GameEnv
import numpy as np
from pyglet.window import key


class Character(GameEnv):
    def __init__(self, char_index: int, game_env: GameEnv, other_chars: np.ndarray, n_rays: int):
        self.char_index = char_index
        self.state = game_env.state
        self.char_width = game_env.char_width
        self.char_height = game_env.char_height
        self.char_state = self._get_index_state(char_index)
        self.char_box = np.array([
            [0, 0],
            [0, self.char_height],
            [self.char_width, 0],
            [self.char_width, self.char_height],
        ]) - (self.char_width // 2)
        self.hitbox_rad = np.sqrt(
            (self.char_width // 2)**2 +
            (self.char_height // 2)**2
        )
        self.direction = np.array([10, 0])
        self.other_chars = other_chars
        self.n_rays = n_rays
        self.key_map = game_env.key_map
        self.anim_map = game_env.anim_map
        self.direct_map = game_env.direct_map
        self.window_height = game_env.window_height
        self.window_width = game_env.window_width

    def update_char_position(self, symbol, state):
        self.state = state
        self.char_state = self._get_index_state(self.char_index)
        if self.char_state['stance'] == 'dead':
            return self.state

        if symbol in self.key_map:
            vector = self.key_map[symbol]
            self.direction = vector
            new_coords, char_collided, map_collided, who_hit = self._propose_movement(
                vector)
            self._set_char_key('coords', new_coords)
            if map_collided:
                self._set_char_key('stance', 'dead')
                return self.state
            elif char_collided:
                self._set_char_key('stance', 'jumping_' +
                                   self.anim_map[symbol])
            else:
                self._set_char_key('stance', 'standing_' +
                                   self.anim_map[symbol])
        elif symbol == key.SPACE:
            self._set_char_key(
                'stance',
                'attack_' + self.direct_map[tuple(self.direction)]
            )
            new_coords, char_collided, map_collided, who_hit = self._propose_movement(
                self.direction)
            if char_collided:
                for who in who_hit:
                    self.state['stance'][who] = 'dead'
                self._set_char_key(
                    'coords',
                    new_coords
                )
            elif map_collided:
                self._set_char_key(
                    'coords',
                    self.state['coords'][self.char_index] - self.direction
                )
                self._set_char_key('stance', 'dead')
                return self.state
            else:
                self._set_char_key(
                    'coords',
                    new_coords
                )
        else:
            print('Invalid symbol')
        return self.state

    def get_vision(self):
        return self._send_rays(5, self.n_rays)

    def _propose_movement(self, vector):
        new_coords = self.state['coords'][self.char_index] + vector
        map_collided = self._map_collision(new_coords)
        char_collided, who_hit = self._char_collision(new_coords)
        has_collided = map_collided or char_collided
        if has_collided:
            return self.state['coords'][self.char_index], char_collided, map_collided, who_hit
        else:
            return new_coords, char_collided, map_collided, who_hit

    def _map_collision(self, new_coords):
        new_box = self.char_box + new_coords
        return np.any(new_box < 0) or \
            np.any(new_box[:, 0] > self.window_width - 1) or \
            np.any(new_box[:, 1] > self.window_height - 1)

    def _char_collision(self, new_coords):
        distances = np.sqrt(
            ((new_coords-self.state['coords']
              [self.other_chars])**2).sum(axis=1)
        )
        collisions = np.array(distances < self.hitbox_rad)
        return np.any(collisions), self.other_chars[collisions]

    def _get_char_key(self, key):
        return self.state[key][self.char_index]

    def _set_char_key(self, key, value):
        self.state[key][self.char_index] = value

    def _send_rays(self, ray_dist, n_rays):
        center = self.state['coords'][self.char_index]

        def handle_ray(ray_ix):
            ## Returns distance to collision and what it collides with.
            ## Wall = 0 and Character = 1
            angle = ray_ix * np.pi * 2 / n_rays
            #ray_dist = 0.1
            d = np.array([
                ray_dist*np.cos(angle),
                ray_dist*np.sin(angle)
            ])

            def handle_ray_char_collision(other_char):
                f = center - self.state['coords'][other_char]
                a = np.dot(d, d)
                b = 2 * np.dot(f, d)
                c = np.dot(f, f) - (self.hitbox_rad ** 2)
                det = b**2 - 4*a*c
                crosses = det >= 0
                if crosses:
                    t_0 = (np.sqrt(det) - b) / (2*a)
                    t_1 = -1 * (b + np.sqrt(det)) / (2*a)
                    if np.sign(t_0) == 1 and np.sign(t_1) == -1:
                        return t_0, crosses
                    elif np.sign(t_0) == -1 and np.sign(t_1) == 1:
                        return t_0, crosses
                    elif np.sign(t_0) == 1 and np.sign(t_1) == 1:
                        return min(t_0, t_1), crosses
                    else:
                        return None, False
                else:
                    return None, crosses

            ray_char_collisions = [
                handle_ray_char_collision(other_char)
                for other_char in self.other_chars
            ]
            ray_char_collisions = [
                collision for collision in ray_char_collisions
                if collision[1]
            ]
            if len(ray_char_collisions) > 0:
                scaler = min(ray_char_collisions, key=lambda x: x[0])[0]
                return scaler * np.linalg.norm(d), 1

            def handle_wall(coll_coord, coll_coord_ix, max_lim):
                coll_other_ix = np.abs(coll_coord_ix - 1)
                t = (coll_coord - center[coll_coord_ix]) / d[coll_coord_ix]
                if t < 0:
                    return None, False
                coll_other = center[coll_other_ix] + (t * d[coll_other_ix])
                if 0 <= coll_other <= max_lim:
                    dist = np.linalg.norm(t*d)
                    if coll_coord_ix == 0:
                        dist_norm = dist / self.window_width
                    else:
                        dist_norm = dist / self.window_height
                    return dist_norm, True
                return None, False

            collisions = {
                'left': handle_wall(0, 0, self.window_height),
                'right': handle_wall(self.window_width, 0, self.window_height),
                'down': handle_wall(0, 1, self.window_width),
                'up': handle_wall(self.window_height, 1, self.window_width),
            }
            collisions = [
                (value[0], key)
                for key, value
                in collisions.items()
                if value[1] == True
            ]

            try:
                return min(collisions, key=lambda x: x[0])[0], 0
            except:
                print(self.state['coords'][self.char_index])
                print(self.char_width)
                print(self.char_height)
                print(ray_ix)
                print(angle)
                raise

        return self._flatten([
            handle_ray(ray_ix)
            for ray_ix in range(n_rays)
        ])

    def _get_index_state(self, index):
        return {
            key: self.state[key][index]
            for key in self.state
        }

    def _flatten(self, l):
        return [item for sublist in l for item in sublist]
