from dataclasses import dataclass
import datetime
import gymnasium as gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dqn.dqn import DQN
from dqn.agent import DQNAgent 
from dqn.memory import ReplayMemory, Transition
from dqn.env import preprocess_observation

@dataclass
class TrainingConfig:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nb_episodes: int = 1000
    nb_steps_per_episode: int = 10000
    memory_capacity: int = 100000
    batch_size: int = 32
    train_frequency: int = 4
    lr: float = 0.00025
    gamma: float = 0.99
    agent_epsilon_start: float = 1.0
    agent_epsilon_decay: int = 500000
    agent_epsilon_end: float = 0.1
    target_update_frequency: int = 1000
    checkpoint_frequency: int = 100

def training(env: gym.Env, config: TrainingConfig = TrainingConfig()):
    
    # Unpack config
    device = config.device
    print(f"Training on device: {device}")

    nb_episodes = config.nb_episodes
    nb_steps_per_episode = config.nb_steps_per_episode
    memory_capacity = config.memory_capacity
    batch_size = config.batch_size
    train_frequency = config.train_frequency
    lr=config.lr
    gamma = config.gamma
    target_update_frequency = config.target_update_frequency
    checkpoint_frequency = config.checkpoint_frequency

    # Initialization
    q_network = DQN(env.action_space.n).to(device)
    target_network = DQN(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    agent = DQNAgent(q_network, env.action_space.n, epsilon_start=config.agent_epsilon_start, epsilon_decay=config.agent_epsilon_decay, epsilon_end=config.agent_epsilon_end, device=device)
    memory = ReplayMemory(memory_capacity)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    writer = SummaryWriter(f"runs/dqn_experiment_{int(datetime.datetime.now().timestamp())}")

    step_count = 0
    # progress_bar = tqdm.tqdm(total=nb_episodes, desc="Training Episodes")
    for episode in range(1, nb_episodes + 1):
        
        state, _ = env.reset()
        state = preprocess_observation(state)
        
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0

        for t in range(nb_steps_per_episode):
            step_count += 1
            # Select and perform an action
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Store the transition in memory
            next_state = None if done else preprocess_observation(next_state)
            memory.push(state, action, reward, next_state)

            state = next_state

            # Training step
            if len(memory) >= batch_size and t % train_frequency == 0:

                batch = memory.sample(batch_size)

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=agent.device,
                    dtype=torch.bool,
                )

                # Prepare batches
                state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
                # Build a tensor only for non-final next states (preserve ordering)
                if non_final_mask.any():
                    non_final_next_states = torch.from_numpy(
                        np.stack([s for s in batch.next_state if s is not None])
                    ).float().to(device)
                else:
                    non_final_next_states = None

                action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)

                # Compute Q(s_t, a)
                q_values = q_network(state_batch)
                q_value = q_values.gather(1, action_batch)

                # Compute Target Q values
                next_q_values = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    if non_final_mask.any():
                        next_q_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]
                q_next = reward_batch + gamma * next_q_values

                loss = criterion(q_value.squeeze(), q_next)
            
                if t % 100 == 0:
                    writer.add_scalar("Q-Values/Mean", q_values.mean().item(), episode * config.nb_steps_per_episode + t)
                    writer.add_scalar("Q-Values/Std", q_values.std().item(), episode * config.nb_steps_per_episode + t)
                    # print(f"Épisode {episode}, Step {t}: Q={q_values.mean().item():.1f}±{q_values.std().item():.1f}, "f"Loss={loss.item():.3f}, +1s={(reward_batch==1).sum().item()}")
             
                # Update Q-Network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loss_count += 1
            
            # Update target network
            if step_count % target_update_frequency == 0:
                print(f"Updating target network at step {step_count}")
                target_network.load_state_dict(q_network.state_dict())

            if done:
                break
        
        # Logging
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            writer.add_scalar("Loss/Training", avg_loss, episode)
            writer.add_scalar("Reward/Training", total_reward, episode)
            writer.add_scalar("Exploration/ε", agent.epsilon, episode)
            print(f"Episode {episode} - Total Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")

        # Save checkpoint
        if episode % checkpoint_frequency == 0:
            torch.save(q_network.state_dict(), f"checkpoints/dqn_pong_{episode}.pth")

        # progress_bar.update(1)
    # progress_bar.close()
    
    env.close()
    writer.close()

def evaluate(env: gym.Env, agent: DQNAgent, n_episodes: int = 5) -> float:
    total_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            action = agent.get_best_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(total_reward)
    return np.mean(total_rewards)
