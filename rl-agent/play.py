import gym
import gym_fighters
import numpy as np
from model import Model
import time

def main(n_chars=2, n_rays=16):
    chars = range(n_chars)
    env = gym.make('fighters-v0', num_chars=n_chars, n_rays=n_rays)
    models = [
        Model(env.observations_dim, env.action_choices, 1)
        for i in range(n_chars)
    ]
    while True:
        state = env.reset()
        done=False
        while not done:
            for char_ix in chars:
                env.render()
                time.sleep(.01)
                action = models[char_ix].predict(state[char_ix, :], explore=False)
                new_state, _, done, _ = env.step(action, char_ix)
                state = new_state
                if done:
                    break

            print(done)
            #state = new_state


if __name__ == "__main__":
    main()
