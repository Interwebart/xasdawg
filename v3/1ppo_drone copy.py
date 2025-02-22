# Файл: ppo_drone.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from drone_env import DroneEnv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Исправлено!

    def forward(self, state):
        return self.critic(state)

    def act(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state)
        return action, log_prob, value.squeeze()

class PPOBuffer:
    def __init__(self, capacity):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.capacity = capacity

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.batch_size = 64
        self.update_epochs = 10
        
        self.buffer = PPOBuffer(2048)

    def update(self):
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.buffer.values)).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer.dones)).to(self.device)
        
        # Расчет преимуществ
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Обновление политики
        for _ in range(self.update_epochs):
            new_values = self.policy(states).squeeze()
            action, new_log_probs, _ = self.policy.act(states)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (new_values - (rewards + advantages)).pow(2).mean()
            entropy_loss = -0.01 * new_log_probs.mean()
            
            total_loss = actor_loss + critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.buffer.clear()

def train():
    env = DroneEnv()
    state_dim = 9
    action_dim = 4
    
    agent = PPO(state_dim, action_dim)
    max_episodes = 500
    max_steps = 200
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).to(agent.device)
            
            with torch.no_grad():
                action, log_prob, value = agent.policy.act(state_tensor)
            
            next_state, reward, done, _, _ = env.step(action.cpu().numpy())
            
            agent.buffer.store(
                state, 
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                reward,
                value.item(),
                done
            )
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        if len(agent.buffer.rewards) >= agent.batch_size:
            agent.update()
        
        print(f"Episode {episode+1}/{max_episodes} | Reward: {episode_reward:.1f}")
        
        if (episode+1) % 50 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_drone_{episode+1}.pth")
    
    env.close()

if __name__ == "__main__":
    train()