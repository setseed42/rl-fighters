import tensorflow as tf
from tensorflow import keras
import os
import datetime

#keras.backend.set_floatx('float64')
class Model(object):
    def __init__(self, observation_dim, actions_dim, i):
        tf.random.set_seed(i)
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
            self.model = self._model_arch(observation_dim, actions_dim)

        self.epoch = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/gradient_tape/{current_time}/char_{i}'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

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
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss(loss)
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
    def predict(self, x):
        return self.model(x)
