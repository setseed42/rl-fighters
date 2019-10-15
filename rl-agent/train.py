import numpy as np
import gym
from model import Model
import gym_fighters
import time

def replace_models_with_model(i, models):
    model = models[i]
    for (j, other_model) in enumerate(models):
        if j!=i:
            other_model.replace_model(model.model)
    return models

def play_game(env, models, render=False):
    steps = 0
    n_chars = env.num_chars
    chars = range(n_chars)
    state = env.reset()
    done = False

    while not done:
        for char_ix in chars:
            other_char_ixes = [i for i in chars if i != char_ix]
            if render:
                env.render()
                time.sleep(.01)
            action = models[char_ix].predict(state[char_ix, :])
            new_state, reward, done, _ = env.step(action, char_ix)
            models[char_ix].add_to_memory(state[char_ix, :], action, reward[char_ix])
            state = new_state
            steps += 1
            if done:
                if steps == 1:
                    return models
                reward = models[char_ix].train(last=True)
                models = replace_models_with_model(char_ix, models)
                for other_char_ix in other_char_ixes:
                    models[other_char_ix].train(last=False, others_reward=reward)
                    models = replace_models_with_model(other_char_ix, models)
                return models

def get_mean_cause_loss(models):
    return np.array([
        np.mean(model.cause_losses)
        for model in models
    ])

def main(gamma=0.99, n_chars=2, n_rays=16):
    env = gym.make('fighters-v0', num_chars=n_chars, n_rays=n_rays)
    episode_nb = 1
    models = [
        Model(env.observations_dim, env.action_choices, i)
        for i in range(n_chars)
    ]
    epochs_before_saving = 100
    while np.all(get_mean_cause_loss(models) > 0.01) or episode_nb < 100:
        if episode_nb % epochs_before_saving == 0:
            models = play_game(env, models, render=True)
            print(f'Game {episode_nb} done')
            for model in models:
                model.save_model()
        else:
            models = play_game(env, models, render=False)
        episode_nb += 1
        # if episode_nb % 10**3 == 0:
        #     print('Generation!')
        #     good_model = np.argmax(running_reward)
        #     bad_model = np.argmin(running_reward)
        #     models[bad_model].replace_model(models[good_model].model)

    print('Finished!')
    for model in models:
        model.save_model()


if __name__ == "__main__":
    np.random.seed(42)
    main()
