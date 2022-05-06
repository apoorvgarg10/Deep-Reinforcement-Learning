import gym
from gym import wrappers, logger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
from collections import deque
import random
import numpy as np
import math
import cv2

class DQNAgent:
  def __init__(self,
                state_space, 
                action_space, 
                episodes=500):
    """DQN Agent on CartPole-v0 environment
    Arguments:
        state_space (tensor): state space
        action_space (tensor): action space
        episodes (int): number of episodes to train
    """
    self.action_space = action_space
    self.gamma = 1.
    self.EPS_START = 1.
    self.EPS_END = 0.01
    self.EPS_DECAY = 1000
    self.steps_done=0.
    self.FRAME_STEP = 4
    self.ROWS=160
    self.COLS=240
    self.state=torch.zeros((self.FRAME_STEP,self.ROWS,self.COLS))
    self.batch_size=256
    self.n_steps=3
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Q Network for training
    self.q_model = model(action_space.n).to(self.device)
    # target Q Network
    self.target_q_model = model(action_space.n).to(self.device)
    self.max_episode_steps=1000
    self.replay_counter = 0
    self.enable_double_q_learning=False
    self.samples_processed=0
    print(self.device)
    # Alphas of PER
    self.alpha=0.6
    self.memory_frame=1
    self.beta_frames=10000
    self.beta_start=0.4
    

  def send_frame_to_state(self,frame):
    resize = T.Compose([T.ToTensor()])
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
    frame[frame < 255] = 0
    frame = frame / 255
    self.state=torch.roll(self.state, 1, 0)
    self.state[0,:,:]=resize(frame)
    return self.state


  def select_action(self,state):
    # if self.EPS>self.EPS_END:
    #     self.EPS*=(1-self.EPS_DECAY)
    eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
    # print("***Epsilon***",eps_threshold)
    if np.random.rand() < eps_threshold:
      return torch.tensor(self.action_space.sample(),device=self.device)
    state=state.to(self.device)
    q_values=self.q_model(state)
    return(torch.argmax(q_values))

  def run_one_forward_q_model(self,state):
    self.q_model(state)

  def init_replay_buf(self,max_len=5000):
    self.memory=deque([],maxlen=max_len)
    self.priorities=deque(maxlen=max_len)
    
  def get_probs(self):
    scaled_ps=np.array(self.priorities) ** self.alpha
    sample_ps=scaled_ps/sum(scaled_ps)
    return sample_ps
    
  def add_to_mem(self,val):
    self.memory.append(val)
    self.priorities.append(max(self.priorities,default=1))

  def get_mem_size(self):
    return len(self.memory)
  
  def set_loss(self):
    self.loss=nn.MSELoss()
  
  def set_optim(self):
    self.opti=optim.Adam(self.q_model.parameters(),lr=3e-5)

  def extract_batch_from_mem(self):
    if len(self.memory)<self.batch_size:
      return None
    multistep_sample=[]
    beta=min(1.,self.beta_start+self.memory_frame*(1.-self.beta_start)/self.beta_frames)
    self.memory_frame+=1
    self.samples_processed+=self.batch_size
    sample_probs=self.get_probs()
    sample_indices=np.random.choice(range(len(self.memory)), self.batch_size, p=sample_probs)
    for index in sample_indices:
        reward_ms=0
        state,a,next_state,reward=self.memory[index]
        # ms_reward+=reward
        for i in range(self.n_steps):
                if (len(self.memory) <= index+i):
                    break
                s, a, r, ns = self.memory[index+i]
                reward_ms += r
                next_state_ms = ns
                if (ns is None):
                    break
        multistep_sample.append((state,a,next_state_ms, reward_ms))
        
    
    p_min=sample_probs.min()
    max_weight=(p_min*len(self.memory))**(-beta)
    weights=(len(self.memory)*sample_probs[sample_indices])**(-beta)
    weights/=max_weight
    weights = torch.tensor(weights, device=self.device, dtype=torch.float)
    return multistep_sample,sample_indices,weights

  def update_ps(self,indices,ps):
    for idx,p in zip(indices,ps):
        self.priorities[idx]=(p+1e-5)**self.alpha
    
    
  def get_batch(self):
    tup=self.extract_batch_from_mem()
    if tup==None:
      return
    samples,ids,weights=tup
    # state,a,next_state,reward=batch
    # print(state.shape,a.shape,next_state.shape,reward.shape)
    states=[]
    actions=[]
    next_states=[]
    valid_ns_indices=[]
    rewards=[]
    
    for i,item in enumerate(samples):
      state,a,next_state,reward,=item
      states.append(state)
      actions.append(a)
      rewards.append(reward)
      if next_state==None:
        continue
      valid_ns_indices.append(torch.tensor(i))
      next_states.append(next_state)
    states=torch.stack(states,dim=0)
    actions=torch.stack(actions,dim=0)
    next_states=torch.stack(next_states,dim=0)
    rewards=torch.stack(rewards,dim=0)
    valid_ns_indices=torch.stack(valid_ns_indices,dim=0)
    # print('states,actions,next_states,rewards',states.shape,actions.shape,next_states.shape,rewards.shape)
    return states,actions,next_states,rewards,valid_ns_indices,ids,weights

  def train_q_model(self):
    if self.get_batch() is None:
      return
    states,actions,next_states,rewards,valid_ns_indices, ids, weights=self.get_batch()
    states=states.to(self.device)
    next_states=next_states.to(self.device)
    rewards=rewards.to(self.device)
    q_state=self.q_model(states).gather(1,actions.unsqueeze(1))
    q_next_state_model=self.q_model(next_states)
    q_next_state=torch.zeros(self.batch_size,device=self.device)
    # q_next_state[valid_ns_indices]=torch.max(self.target_q_model(next_states))
    q_next_state_target_model=self.target_q_model(next_states)
    # print('shape_target model:: ',q_next_state_target_model.shape)
    argmax=torch.argmax(q_next_state_model)
    # print('indices shape:: ',valid_ns_indices.shape)
    # print('target model out shape::',q_next_state_target_model.shape)
    # print('q model out shape::',q_next_state_model.shape)
    if self.enable_double_q_learning:
        # print('Double deep Q Learning')
        q_next_state[valid_ns_indices]=q_next_state_target_model.gather(1,q_next_state_model.max(1)[1].detach().unsqueeze(1)).squeeze(1)
    else:
        # print('Normal deep Q Learning')
        q_next_state[valid_ns_indices] = q_next_state_target_model.max(1)[0].detach()
    # print('q values comparison',q_state,q_next_state)
    yj=q_next_state*self.gamma+rewards
    # print(q_state.squeeze(1).shape,yj.shape)
    delta=q_state.squeeze()-yj
    self.opti.zero_grad()
    loss=(0.5*delta*delta*weights).mean()
    # loss=self.loss(q_state,yj.unsqueeze(1))
    # print(loss)
    loss.backward()
    # for param in self.target_q_model.parameters():
    #   print(param)
    td_error=delta.abs().detach().cpu().numpy().tolist()
    self.update_ps(ids,td_error)
    for param in self.q_model.parameters():
      param.grad.data.clamp_(-1, 1)
    self.opti.step()    

# Model 
class model(nn.Module):
  def __init__(self,action_space):
    super(model,self).__init__()
    # print(action_space)
    self.conv1=nn.Conv2d(4,64,kernel_size=5,stride=3)
    self.bn1=nn.BatchNorm2d(64)
    self.conv2=nn.Conv2d(64,64,kernel_size=4,stride=2)
    self.bn2=nn.BatchNorm2d(64)
    self.conv3=nn.Conv2d(64,64,kernel_size=3,stride=1)
    self.bn3=nn.BatchNorm2d(64)
    self.relu=nn.ReLU()
    self.linear1=nn.Linear(52992,512)
    self.linear2=nn.Linear(512,256)
    self.linear3=nn.Linear(256,64)
    # self.linear4=nn.Linear(64,2)
    self.adv=nn.Linear(64,2)
    self.val=nn.Linear(64,1)
    self.flatten=nn.Flatten()
  def forward(self,x):
    # print(x.shape)
    x=self.conv1(x)
    x=self.bn1(x)
    x=self.relu(x)
    # print(x.shape)
    x=self.conv2(x)
    x=self.bn2(x)
    x=self.relu(x)
    # print(x.shape)
    x=self.conv3(x)
    x=self.bn3(x)
    x=self.relu(x)
    # print(x.shape)
    x=self.flatten(x)
    # print(x.shape)
    x=self.relu(self.linear1(x))
    x=self.relu(self.linear2(x))
    x=self.relu(self.linear3(x))
    adv=self.relu(self.adv(x))
    val=self.relu(self.val(x))
    # print(x.shape)
    # return x
    return val+adv-val.mean()


def render_frame_wo_step(env):
  return env.render(mode='rgb_array')


if __name__ == '__main__':
    env_id = "CartPole-v0"
    no_render = False
    
    # the number of trials without falling over
    win_trials = 100
    # the CartPole-v0 is considered solved if 
    # for 100 consecutive trials, he cart pole has not 
    # fallen over and it has achieved an average 
    # reward of 195.0 
    # a reward of +1 is provided for every timestep 
    # the pole remains upright
    win_reward = { 'CartPole-v0' : 195.0 }

    env = gym.make(env_id)
    #change the max steps of cartpole to 500, so that the max reward is not capped at 200
    env._max_episode_steps = 500
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.init_replay_buf(10000)
    agent.set_loss()
    agent.set_optim()
    agent.target_q_model.load_state_dict(agent.q_model.state_dict())
    array_avg_rewards=[]
    agent.enable_double_q_learning=True
    # screen_np,screen_tensor=fill_init_states(env)
    # print(screen_np.shape,screen_tensor.shape)
    array_total_rewards=[]
    for i in range(3000):
      agent.steps_done+=1
      env.reset()
      for frame_count in range(agent.FRAME_STEP):
        frame=render_frame_wo_step(env)
        agent.send_frame_to_state(frame)
      total_reward=0
      n_step=0
      while True:
        
        a=agent.select_action(agent.state.unsqueeze(0))
        ns,reward,done,_=env.step(a.item())
        total_reward+=reward
        frame=render_frame_wo_step(env)
        state=agent.state
        next_state=agent.send_frame_to_state(frame)
        # if not done or n_step>=agent.max_episode_steps:
        #     reward=reward
        # else:
        #     reward=-100   
        # if done:
        #   next_state=None
        if done:
          next_state=None
        agent.add_to_mem((state,a,next_state,torch.tensor(reward)))
        # train the model here
        # if (next_state!=None and torch.equal(state,next_state)):
        #   print('state and next_state equal')
        agent.train_q_model()
        if done:
          break
        # print(agent.get_mem_size())
        n_step+=1
      array_total_rewards.append(total_reward)
      avg_reward=sum(array_total_rewards[-100:])/len(array_total_rewards[-100:])
      array_avg_rewards.append(avg_reward)
      if i%50==0:
        print('***Episode Number***',i)
        print('total reward:',total_reward)
        print('average reward:',avg_reward)
      if i%10==0:
        # print('Updating target model!')
        agent.target_q_model.load_state_dict(agent.q_model.state_dict())
      if avg_reward>195.:
        print('****SOLVED, max reward reached!!!****')
        break
      # if i%1000==0:
      #   print('****Episode no: *****',i)
      #   print('total rewards::',array_total_rewards)
      #   print('100 iteration average rewards::',array_avg_rewards)
    print('total rewards::',array_total_rewards)
    print('100 iteration average rewards::',array_avg_rewards)
    torch.save(agent.target_q_model.state_dict(), 'models/agent_target_q_model_per_dqn_dueling.pth')
    torch.save(agent.q_model.state_dict(), 'models/agent_q_model_per_dqn_dueling.pth')
    print('******************* Total Images Processed *************************', agent.samples_processed)
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.plot(array_avg_rewards)
    plt.savefig('rewards_avg_100_per_dqn_dueling.png')
    
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.plot(array_total_rewards)
    plt.savefig('rewards_per_dqn_dueling.png')
