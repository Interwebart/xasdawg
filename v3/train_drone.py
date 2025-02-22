# Файл dqn_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from drone_env import DroneEnv
import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class FlightWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FlightWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(6)
        
    def _discrete_to_continuous(self, action):
        action_map = {
            0: [0.2, 0.0, 0.0, 0.0],
            1: [-0.2, 0.0, 0.0, 0.0],
            2: [0.0, 0.2, 0.0, 0.0],
            3: [0.0, -0.2, 0.0, 0.0],
            4: [0.0, 0.0, 0.2, 0.0],
            5: [0.0, 0.0, -0.2, 0.0]
        }
        return np.array(action_map[action], dtype=np.float32)

    def step(self, action):
        continuous_action = self._discrete_to_continuous(action)
        return super().step(continuous_action)

def train():
    env = FlightWrapper(DroneEnv())
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    episodes = 200
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        if episode % 10 == 0:
            agent.update_target()
            print(f"Episode: {episode+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    torch.save(agent.policy_net.state_dict(), "dqn_drone.pth")
    env.close()

if __name__ == "__main__":
    train()