import tensorflow as tf
from tensorflow import keras
import os
import datetime
import numpy as np
from collections import deque
from abc import ABCMeta, abstractmethod
# keras.backend.set_floatx('float64')


class AbstractModel(metaclass=ABCMeta):
    def __init__(self, observation_dim, actions, n_chars, name, batch_size, shuffle, char_ix):
        tf.random.set_seed(42)
        self.actions = actions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model_path = f'./model/char_{name}'
        self.model_name = f'{n_chars}_model.hdf5'
        self.full_model_path = f'{self.model_path}/{self.model_name}'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if os.path.exists(self.full_model_path):
            print("loading previous weights")
            self.model = tf.keras.models.load_model(
                self.full_model_path,
            )
        else:
            print('Training from scratch')
            self.model = self._model_arch(observation_dim, len(self.actions))

        self.epoch = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/gradient_tape/{current_time}/char_{name}_{n_chars}_{char_ix}'
        self.train_summary_writer = tf.summary.create_file_writer(
            train_log_dir)
        self.exploration = 1
        self.gamma = 0.99
        self.reset_memory()
        self.min_exploration = 0.01
        self.exploration_decay = 0.001
        self.cause_wins = deque(maxlen=100)
        self.cause_losses = deque(maxlen=100)
        self.running_reward = None

    @abstractmethod
    def _model_arch(self):
        pass

    @abstractmethod
    def end_game(self):
        pass

    def train_loop(self, sample_weights, cause_wins, cause_losses, steps):
        history = self.model.fit(
            np.array(self.memory['x']), np.array(self.memory['y']),
            sample_weight=np.array(sample_weights) if len(sample_weights)>1 else None,
            epochs=1,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            verbose=0
        )
        with self.train_summary_writer.as_default():
            for key, items in history.history.items():
                tf.summary.scalar(key, items[-1], step=self.epoch)
            tf.summary.scalar('running_reward',
                              self.running_reward, step=self.epoch)
            tf.summary.scalar('exploration', self.exploration, step=self.epoch)
            tf.summary.scalar('cause_wins', cause_wins, step=self.epoch)
            tf.summary.scalar('cause_losses', cause_losses, step=self.epoch)
            tf.summary.scalar('steps', len(self.memory['x']), step=self.epoch)
        self.end_game()
        self.epoch += 1

    def save_model(self):
        self.model.save(self.full_model_path)

    def replace_model(self, model):
        self.model = model

    def get_running_reward(self):
        if self.running_reward is None:
            return self.memory['reward_sum']
        else:
            return self.running_reward * 0.99 + self.memory['reward_sum'] * 0.01

    def train(self, last, others_reward=None):
        if last:
            if self.memory['reward_sum'] == 1:
                # Means character killed other player
                self.memory['reward'][-1] *= 1
                # maybe scale up reward sum as well
                self.cause_wins.append(1)
                self.cause_losses.append(0)
            if self.memory['reward_sum'] == -1:
                # Means character hit wall
                self.cause_losses.append(1)
                self.cause_wins.append(0)
            reward = self.memory['reward'][-1]
        else:
            reward = others_reward * -1
            self.cause_wins.append(0)
            self.cause_losses.append(0)
            self.memory['reward'][-1] = reward
            self.memory['reward_sum'] += reward
        self.running_reward = self.get_running_reward()
        self.train_loop(
            sample_weights=self.discount_rewards(
                self.memory['reward'], self.gamma),
            cause_wins=np.mean(self.cause_wins),
            cause_losses=np.mean(self.cause_losses),
            steps=len(self.memory['x'])
        )
        self.exploration = max(
            self.min_exploration,
            self.exploration*(1-self.exploration_decay)
        )
        self.reset_memory()
        return reward

    def predict(self, x, explore=True):
        if explore:
            exploration = self.exploration
        else:
            exploration = 0
        x = np.expand_dims(x, axis=0)
        p_exp = np.random.uniform(0, 1, 1)[0]
        if p_exp < exploration:
            return np.random.choice(self.actions, 1)[0]
        else:
            action_proba = self.model.predict(x, verbose=0)[0]
            return np.random.choice(self.actions, 1, p=action_proba)[0]

    def add_to_memory(self, x, y, reward):
        self.memory['x'].append(x)
        y_vec = np.zeros(len(self.actions))
        y_vec[self.actions.index(y)] = 1
        self.memory['y'].append(y_vec)
        self.memory['reward'].append(reward)
        self.memory['reward_sum'] += reward

    def reset_memory(self):
        self.memory = {
            'x': [],
            'y': [],
            'reward': [],
            'reward_sum': 0
        }
    # Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)

    def discount_rewards(self, r, gamma):
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


class MLPModel(AbstractModel):
    def __init__(self, observation_dim: int, actions: np.ndarray, n_chars: int, char_ix=0):
        super().__init__(observation_dim, actions, n_chars, 'mlp', 32, True, char_ix)

    def _model_arch(self, observation_dim: int, actions_dim: int):
        input_layer = keras.layers.Input(shape=(observation_dim,))
        # x = keras.layers.BatchNormalization()(input_layer)
        x = keras.layers.Dense(200, activation='relu',
                               kernel_initializer='glorot_uniform')(input_layer)
        output_layer = keras.layers.Dense(
            actions_dim, activation='softmax', kernel_initializer='RandomNormal')(x)
        model = keras.models.Model(input_layer, output_layer)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def end_game(self):
        pass


class LSTMModel(AbstractModel):
    def __init__(self, observation_dim, actions, n_chars, char_ix=0):
        super().__init__(observation_dim, actions, n_chars, 'lstm', 1, False, char_ix)

    def _model_arch(self, observation_dim, actions_dim):
        input_layer = keras.layers.Input(
            shape=(observation_dim,), batch_size=1)
        x = keras.layers.BatchNormalization()(input_layer)
        x = keras.layers.RepeatVector(1)(x)
        x = keras.layers.LSTM(
            200, kernel_initializer='glorot_uniform', stateful=True)(x)
        output_layer = keras.layers.Dense(
            actions_dim, activation='softmax', kernel_initializer='RandomNormal')(x)
        model = keras.models.Model(input_layer, output_layer)
        optimizer = keras.optimizers.RMSprop(clipvalue=5)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def end_game(self):
        self.model.reset_states()
