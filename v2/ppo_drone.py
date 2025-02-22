import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from drone_env import DroneEnv, EPISODE_MAX_STEPS

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Актор
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # Критик
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        return self.critic(state)

    def act(self, state):
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state).squeeze()
        return action, log_prob, value

class PPOTrainer:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.batch_size = 512
        self.buffer = deque(maxlen=4096)
        self.update_epochs = 10

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
            
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[3] for t in self.buffer])).to(self.device)
        values = torch.FloatTensor(np.array([t[4] for t in self.buffer])).to(self.device)
        dones = torch.FloatTensor(np.array([t[5] for t in self.buffer])).to(self.device)
        
        # Расчет преимуществ
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
        
        # Обновление политики
        for _ in range(self.update_epochs):
            new_values = self.policy(states).squeeze()
            new_actions, new_log_probs, _ = self.policy.act(states)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
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
    agent = PPOTrainer(state_dim=9, action_dim=4)
    episodes = 2000
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(EPISODE_MAX_STEPS):
            state_tensor = torch.FloatTensor(state).to(agent.device)
            
            with torch.no_grad():
                action, log_prob, value = agent.policy.act(state_tensor)
            
            next_state, reward, done, _, _ = env.step(action.cpu().numpy())
            
            agent.buffer.append((
                state,
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                reward,
                value.item(),
                done
            ))
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.update()
        print(f"Эпизод {episode+1}/{episodes} | Награда: {total_reward:.1f}")
        
        if (episode+1) % 50 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_{episode+1}.pth")
    
    env.close()

if __name__ == "__main__":
    train()