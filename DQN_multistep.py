"""
Neil Gutkin
Final Project: Trains a vision-baesd DQN to solve CartPole-v0 problem
Uses forward-view multi-step learning
"""

from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf
import cv2
import gym
import random
from gym import wrappers, logger

class DQNAgent:
    def __init__(self, action_space, n_steps):
        """Vision-based DQN Agent on CartPole-v0 environment
        Arguments:
            action_space (tensor): action space
            n_steps (int): number of look-ahead steps
        """
        self.action_space = action_space

        # experience buffer
        self.memory = deque(maxlen=10000)

        # discount rate
        self.gamma = 0.95

        # initially 100% exploration
        self.epsilon = 1.0
        # iteratively applying decay til 
        # 1% exploration/99% exploitation
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

        # Q Network weights filename
        self.weights_file = 'dqn_cartpole.ckpt'
        
        # Q Network for training
        n_outputs = action_space.n
        self.q_model = self.build_model(n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam(lr=.00025))
        
        # target Q Network
        self.target_q_model = self.build_model(n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0
        self.n_steps = n_steps
        
    def build_model(self, n_outputs):
        """Q Network is 3 conv layers followed by 3 dense layers
        Arguments:
            n_outputs (int): output dim
        Return:
            q_model (Model): DQN
        """
        inputs = Input(shape=(4, 160, 240, ), name='state')
        x = Conv2D(64, 5, strides=(3,3), activation='relu', data_format='channels_first')(inputs)
        x = Conv2D(64, 4, strides=(2,2), activation='relu', data_format='channels_first')(x)
        x = Conv2D(64, 3, strides=(1,1), activation='relu', data_format='channels_first')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(n_outputs, activation='relu')(x)
        
        q_model = Model(inputs, outputs)
        q_model.summary()
        return q_model
        
    def save_weights(self):
        """save Q Network params to a file"""
        self.q_model.save_weights(self.weights_file)
        
    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())
    
    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def remember(self, state, action, reward, new_state, done):
        """store experiences in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
            done (bool): indicates episode conclusion
        """
        item = (state, action, reward, new_state, done)
        self.memory.append(item)

    def act(self, state):
        """eps-greedy policy
        Arguments:
            state (tensor): env state
        Return:
            action (tensor): action to execute
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        action = np.argmax(q_values[0])
        return action
        
    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value
        
    def replay(self, batch_size):
        """experience replay addresses the correlation issue 
            between samples
        Arguments:
            batch_size (int): replay buffer batch 
                sample size
        """
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(list(enumerate(self.memory)), batch_size)
        state_batch, q_values_batch = [], []

        # map model inputs to targets
        for replay_idx, sars in sars_batch:
            state, action, reward, next_state, done = sars

            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            # get n-step reward and state
            n_reward = 0
            n_state = next_state
            for i in range(self.n_steps):
                if (len(self.memory) <= replay_idx+i):
                    break
                s, a, r, ns, d = self.memory[replay_idx+i]
                n_reward += r
                if (d or i == self.n_steps - 1):
                    n_state = ns
                    break

            # get target Q value (q_max(next_state)*gamma + reward)
            # when n > 1, n-step lookahead rewards are summed and n'th state ahead is used
            target = self.get_target_q_value(n_state, n_reward)

            # set Q value corresponding to the action taken
            q_values[0][action] = reward if done else target

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after 
        # every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

def process_image(image):
    # Simple processing: RGB to GRAY and resizing keeping a fixed aspect ratio
    if len(image.shape) == 3:
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (240, 160))
    # Further processing: make the pixel values binary - either white or not
    image[image < 255] = 0
    image = image / 255
    return image

if __name__ == '__main__':
    
    env_id = "CartPole-v0"
    no_render = True

    # the number of trials without falling over
    win_trials = 100

    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = 195.0

    # stores the reward per episode
    # uses deque to trim old scores
    scores = deque(maxlen=win_trials)

    env = gym.make(env_id)

    outdir = "/tmp/dqn-%s" % env_id
    
    if no_render:
        env = wrappers.Monitor(env,
                               directory=outdir,
                               video_callable=False,
                               force=True)
    else:
        env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # instantiate the DQN agent
    n_ahead = 3
    agent = DQNAgent(env.action_space, n_ahead)

    # should be solved in this number of episodes
    episode_count = 10000
    batch_size = 64
    min_history = 1000

    # by default, CartPole-v0 has max episode steps = 200
    # you can use this to experiment beyond 200
    # env._max_episode_steps = 4000

    # Q-Learning sampling and fitting
    for episode in range(episode_count):
        print("episode: ", episode)
        state = env.reset()
        done = False
        total_reward = 0
        
        stack = np.zeros((4, 160, 240))
        first_image = process_image(env.render(mode='rgb_array'))
        # Each network input (state) is 4 frames
        # Use first frame as initial state for all 4 frames
        stack[[0,1,2,3],:,:] = first_image
        stack = np.reshape(stack, [1, 4, 160, 240])

        while not done:
            # Select an action and take a step
            # in CartPole-v0, action=0 is left and action=1 is right
            action = agent.act(stack)
            state, reward, done, _ = env.step(action)
            # Insert new render into stack
            next_stack = np.roll(stack, -1, axis=1)
            next_stack[0,3,:,:] = process_image(env.render(mode='rgb_array'))
            
            # store every experience unit in replay buffer
            agent.remember(stack, action, reward, next_stack, done)
            stack = next_stack
            total_reward += reward


        # Call experience replay
        if len(agent.memory) >= min_history:
            agent.replay(batch_size)

        scores.append(total_reward)
        mean_score = np.mean(scores)

        # Check convergence
        if mean_score >= win_reward and episode >= win_trials:
            print("Solved in episode %d: \
                   Mean survival = %0.2lf in %d episodes"
                  % (episode, mean_score, win_trials))
            print("Epsilon: ", agent.epsilon)
            agent.save_weights()
            break

        # Print current success rate and save weights
        if (episode + 1) % win_trials == 0:
            print("Episode %d: Mean survival = \
                   %0.2lf in %d episodes" %
                  ((episode + 1), mean_score, win_trials))
            agent.save_weights()

    # close the env and write monitor result info to disk
    env.close() 