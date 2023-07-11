import gym
import gym_fighters
from model import LSTMModel, MLPModel
import time


def main(model, chars_trained_on, n_chars=2, n_rays=16):
    chars = range(n_chars)
    env = gym.make('fighters-v0', num_chars=n_chars, n_rays=n_rays)
    models = [
        model(env.observations_dim, env.action_choices, chars_trained_on)
        for i in range(n_chars)
    ]
    while True:
        state = env.reset()
        done = False
        while not done:
            for char_ix in chars:
                env.render()
                time.sleep(.01)
                action = models[char_ix].predict(
                    state[char_ix, :], explore=False)
                new_state, _, done, _ = env.step(action, char_ix)
                state = new_state
                if done:
                    for model in models:
                        model.end_game()
                    break

            # state = new_state


if __name__ == "__main__":
    main(MLPModel, 3)
    main(LSTMModel, 3)
