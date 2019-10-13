import gym
import gym_fighters
import numpy as np
from model import Model
import time

def main(n_chars=4, n_rays=16):
    chars = range(n_chars)
    env = gym.make('fighters-v0', num_chars=n_chars, n_rays=n_rays)
    models = [
        Model(env.observations_dim, len(env.action_choices), 1)
        for i in range(n_chars)
    ]
    while True:
        state = env.reset()
        done=False
        while not done:
            for char_ix in chars:
                env.render()
                time.sleep(.01)
                x = np.expand_dims(state[char_ix, :], axis=0)
                action_proba = models[char_ix].predict(x)[0]
                action = np.random.choice(
                    env.action_choices, 1, p=action_proba.numpy())[0]

                new_state, _, done, _ = env.step(action, char_ix)
                state = new_state
                print(done)
                if done:
                    break

            print(done)
            #state = new_state


if __name__ == "__main__":
    main()
