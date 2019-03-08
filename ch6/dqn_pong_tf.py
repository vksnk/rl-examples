import env_wrappers

import argparse
import time
import numpy as np
import collections

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5
# DEFAULT_ENV_NAME = "BreakoutDeterministic-v4"
# MEAN_REWARD_BOUND = 300.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

def build_dq_net(n_actions):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=8, strides=4, input_shape=(84, 84, 4), activation='relu', data_format = "channels_last"))
    model.add(layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', data_format = "channels_last"))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format = "channels_last"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.ReLU())
    model.add(layers.Dense(n_actions))

    return model

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions, dtype=np.int32), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.float32), np.array(next_states)

@tfe.defun
def get_next_action(net, state):
    # state_a = tf.to_float(tf.expand_dims(state, axis=0)) / 255.0
    state_a = tf.expand_dims(state, axis=0)
    q_vals_v = net(state_a)
    return tf.argmax(q_vals_v, axis = 1)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = get_next_action(net, self.state).numpy()[0]

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, not is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

@tfe.defun
def calc_loss(net, target_net, states, actions, rewards, dones, next_states):
    with tf.GradientTape() as tape:
        action_row_indices_v = tf.range(tf.shape(actions)[0])
        actions_v = tf.stack([action_row_indices_v, actions], axis=1)

        # next_state_values = tf.reduce_max(target_net(tf.to_float(next_states) / 255.0), axis=1)
        next_state_values = tf.reduce_max(target_net(next_states), axis=1)
        expected_state_action_values = dones * next_state_values * GAMMA + rewards

        # state_action_v = net(tf.to_float(states) / 255.0)
        state_action_v = net(states)
        state_action_v = tf.gather_nd(state_action_v, actions_v)

        loss_value = tf.reduce_mean(tf.squared_difference(state_action_v, expected_state_action_values))
        # loss_value = tf.losses.huber_loss(state_action_v, expected_state_action_values)

    return tape.gradient(loss_value, net.trainable_variables) 

def play_and_visualize_game(env):
    state = env.reset()
    is_done = False

    FPS = 25
    while not is_done:
        start_ts = time.time()
        env.render()
        if True:
            action = env.action_space.sample()
        else:
            state_a = tf.convert_to_tensor([self.state])
            q_vals_v = net(state_a)
            act_v = tf.argmax(q_vals_v, axis = 1)
            action = act_v.numpy()[0]

        state, reward, is_done, _ = env.step(action)
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()

    env = env_wrappers.make_env(args.env)

    net = build_dq_net(env.action_space.n)
    target_net = tf.keras.models.clone_model(net)
    target_net.set_weights(net.get_weights())

    print(net.summary())
    print(net.inputs)

    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer("./tb/")
    writer.set_as_default()

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    speed = 0
    mean_reward = 0

    device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    print("Using device: %s" % device)
    with tf.device(device):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        while True:
            with tf.device('/cpu:0'):
                global_step.assign_add(1)
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

            reward = agent.play_step(net, epsilon)

            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), mean_reward, epsilon,
                    speed
                ))
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    net.save(args.env + "-best.dat")
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                if mean_reward > args.reward:
                    print("Solved in %d frames!" % frame_idx)
                    break
            # continue
            with tf.contrib.summary.record_summaries_every_n_global_steps(5000):
                tf.contrib.summary.scalar("epsilon", epsilon, step=global_step)
                tf.contrib.summary.scalar("speed", speed, step=global_step)
                tf.contrib.summary.scalar("reward_100", mean_reward, step=global_step)
                # tf.contrib.summary.scalar("reward", reward, step=global_step)

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                del target_net
                target_net = tf.keras.models.clone_model(net)
                target_net.set_weights(net.get_weights())

            states, actions, rewards, dones, next_states = buffer.sample(BATCH_SIZE)
            grads = calc_loss(net, target_net, states, actions, rewards, dones, next_states)
            optimizer.apply_gradients(zip(grads, net.trainable_variables), global_step=global_step)
