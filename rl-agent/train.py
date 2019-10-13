import os
import datetime
import numpy as np
import gym
import random
from model import Model
import gym_fighters
from collections import deque
import time

# Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    # we go from last reward to first one so we don't have to do exponentiations
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # if the game ended (in Pong), reset the reward sum
            running_add = 0
        # the point here is to use Horner's method to compute those rewards efficiently
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r -= np.mean(discounted_r)  # normalizing the result
    discounted_r /= np.std(discounted_r)  # idem
    return discounted_r


def init_memory(n_chars):
    x_train = [
        [] for _ in range(n_chars)
    ]
    y_train = [
        [] for _ in range(n_chars)
    ]
    rewards = [
        [] for _ in range(n_chars)
    ]
    reward_sum = np.zeros(n_chars)
    return x_train, y_train, rewards, reward_sum


def play_game(
    env, models,
    running_reward,
    render=False,
    exploration=0,
    cause_wins=[deque(maxlen=100), deque(maxlen=100)],
    cause_losses=[deque(maxlen=100), deque(maxlen=100)]
):
    action_space = env.action_choices
    steps = 0
    n_chars = env.num_chars
    chars = range(n_chars)
    state = env.reset()
    done = False
    x_train, y_train, rewards, reward_sum = init_memory(n_chars)

    while not done:
        for char_ix in chars:
            other_char_ixes = [i for i in chars if i != char_ix]
            if render:
                env.render()
                time.sleep(.01)
            x = np.expand_dims(state[char_ix, :], axis=0)
            p_exp = np.random.uniform(0, 1, 1)[0]
            if p_exp < exploration:
                action = np.random.choice(env.action_choices, 1)[0]
            else:
                try:
                    action_proba = models[char_ix].predict(x)[0]
                    action = np.random.choice(
                        env.action_choices, 1, p=action_proba.numpy())[0]
                except:
                    print(x)
                    print(action_proba)
                    print(state)
                    raise

            y = np.zeros(len(action_space))
            y[action_space.index(action)] = 1
            new_state, reward, done, _ = env.step(action, char_ix)
            y_train[char_ix].append(y)
            x_train[char_ix].append(x)
            rewards[char_ix].append(reward[char_ix])
            reward_sum[char_ix] += reward[char_ix]
            state = new_state
            steps += 1
            if done:
                char_reward_sum = reward_sum[char_ix]
                if char_reward_sum == 1:
                    #Means character killed other player
                    rewards[char_ix][-1] *= 1
                    cause_wins[char_ix].append(1)
                    cause_losses[char_ix].append(0)
                if char_reward_sum == -1:
                    #Means character hit wall
                    cause_losses[char_ix].append(1)
                    cause_wins[char_ix].append(0)
                for other_char_ix in other_char_ixes:
                    #Give other characters inverse of other reward
                    cause_losses[other_char_ix].append(0)
                    cause_wins[other_char_ix].append(0)
                    rewards[other_char_ix][-1] = rewards[char_ix][-1] * -1
                    reward_sum[other_char_ix] += rewards[other_char_ix][-1]


                for char in chars:
                    if running_reward[char] is None:
                        running_reward[char] = reward_sum[char]
                    else:
                        running_reward[char] = running_reward[char] * \
                            0.99 + reward_sum[char] * 0.01

                return {
                    'train_ix': char_ix,
                    'x_train': x_train,
                    'y_train': y_train,
                    'cause_wins': cause_wins,
                    'cause_losses': cause_losses,
                    'exploration': exploration,
                    'rewards': rewards,
                    'running_reward': running_reward,
                    'steps': steps,
                }


def get_mean_cause(which, results):
    return np.array([
        np.mean(cause_loss)
        for cause_loss in results[which]
    ])

def main(gamma=0.99, n_chars=2, n_rays=16):
    env = gym.make('fighters-v0', num_chars=n_chars, n_rays=n_rays)
    episode_nb = 1
    running_reward =[None for _ in range(n_chars)]
    models = [
        Model(env.observations_dim, len(env.action_choices), i)
        for i in range(n_chars)
    ]

    epochs_before_saving = 100
    min_exploration = 0.1
    exploration_decay = 0.001
    init_exploration = 1
    ##Play first game
    results = play_game(
        env, models,
        running_reward,
        exploration=init_exploration,
    )
    while np.all(get_mean_cause('cause_losses', results) > 0.01) or episode_nb < 100:
        for (i, model) in enumerate(models):
            if len(results['rewards'][i]) == 1:
                continue
            model.train_loop(
                x=np.vstack(results['x_train'][i]),
                y=np.vstack(results['y_train'][i]),
                sample_weights=discount_rewards(results['rewards'][i], gamma),
                running_reward=results['running_reward'][i],
                exploration=results['exploration'],
                cause_wins=np.mean(results['cause_wins'][i]),
                cause_losses=np.mean(results['cause_losses'][i]),
                steps=results['steps'],
            )
            for (j, other_model) in enumerate(models):
                if j != i:
                    other_model.replace_model(model.model)
        if episode_nb % epochs_before_saving == 0:
            results = play_game(
                env, models,
                results['running_reward'],
                render=True,
                exploration=max(min_exploration, results['exploration']*(1-exploration_decay)),
                cause_wins=results['cause_wins'],
                cause_losses=results['cause_losses']
            )
            print(f'Game {episode_nb} done')
            for model in models:
                model.save_model()
        else:
            results = play_game(
                env, models,
                results['running_reward'],
                exploration=max(min_exploration,results['exploration']*(1-exploration_decay)),
                cause_wins=results['cause_wins'],
                cause_losses=results['cause_losses']
            )
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
