import tensorflow as tf
from tensorflow import keras
import os
import datetime
import numpy as np
from collections import deque
#keras.backend.set_floatx('float64')
class Model(object):
    def __init__(self, observation_dim, actions, i):
        tf.random.set_seed(i)
        self.actions = actions
        self.model_path = f'./model/char_{i}'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if tf.saved_model.contains_saved_model(self.model_path):
            print("loading previous weights")
            self.model = tf.keras.models.load_model(
                self.model_path,
            )
        else:
            print('Training from scratch')
            self.model = self._model_arch(observation_dim, len(self.actions))

        self.epoch = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/gradient_tape/{current_time}/char_{i}'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.exploration = 1
        self.gamma = 0.99
        self.reset_memory()
        self.min_exploration = 0.01
        self.exploration_decay = 0.001
        self.cause_wins = deque(maxlen=100)
        self.cause_losses = deque(maxlen=100)
        self.running_reward = None

    def _model_arch(self, observation_dim, actions_dim):
        input_layer = keras.layers.Input(shape=(observation_dim,))
        #x = keras.layers.BatchNormalization()(input_layer)
        x = keras.layers.Dense(200, activation='relu',
                               kernel_initializer='glorot_uniform')(input_layer)
        output_layer = keras.layers.Dense(
            actions_dim, activation='softmax', kernel_initializer='RandomNormal')(x)
        return keras.models.Model(input_layer, output_layer)

    def train_loop(self, x, y, sample_weights, running_reward, exploration, cause_wins, cause_losses, steps):
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x, y, sample_weights))
        train_dataset = train_dataset.shuffle(60000).batch(32)
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')

        def train_step(model, optimizer, x_train, y_train, sample_weight):
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = loss_object(y_train, predictions, sample_weight)
                log_loss = loss_object(y_train, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss(log_loss)
            train_accuracy(y_train, predictions)

        for (x_train, y_train, sample_weight) in train_dataset:
            train_step(self.model, optimizer, x_train, y_train, sample_weight)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=self.epoch)
            tf.summary.scalar(
                'accuracy', train_accuracy.result(), step=self.epoch)
            tf.summary.scalar('running_reward',
                              running_reward, step=self.epoch)
            tf.summary.scalar('exploration', exploration, step=self.epoch)
            tf.summary.scalar('cause_wins', cause_wins, step=self.epoch)
            tf.summary.scalar('cause_losses', cause_losses, step=self.epoch)
            tf.summary.scalar('steps', steps, step=self.epoch)

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        self.epoch += 1

    def save_model(self):
        tf.keras.models.save_model(
            self.model,
            self.model_path,
            overwrite=True,
        )
        # tf.saved_model.save(self.model, self.model_path)
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
                #Means character killed other player
                self.memory['reward'][-1] *= 1
                #maybe scale up reward sum as well
                self.cause_wins.append(1)
                self.cause_losses.append(0)
            if self.memory['reward_sum'] == -1:
                #Means character hit wall
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
            x=self.memory['x'],
            y=self.memory['y'],
            sample_weights=self.discount_rewards(self.memory['reward'], self.gamma),
            running_reward=self.running_reward,
            exploration=self.exploration,
            cause_wins=np.mean(self.cause_wins),
            cause_losses=np.mean(self.cause_wins),
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
            action_proba = self.model.predict(x)[0]
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
