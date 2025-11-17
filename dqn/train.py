from dataclasses import dataclass
import datetime 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from dqn.dqn import DQN
from dqn.agent import DQNAgent 
from dqn.memory import ReplayMemory, Transition

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
    target_network.eval()
    
    agent = DQNAgent(q_network, env.action_space.n, epsilon_start=config.agent_epsilon_start, epsilon_decay=config.agent_epsilon_decay, epsilon_end=config.agent_epsilon_end, device=device)
    memory = ReplayMemory(memory_capacity)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    # Use Huber loss (Smooth L1) which is more stable for Q-learning
    criterion = nn.SmoothL1Loss()
    writer = SummaryWriter(f"runs/dqn_experiment_{int(datetime.datetime.now().timestamp())}")


    # progress_bar = tqdm.tqdm(total=nb_episodes, desc="Training Episodes")
    for episode in range(1, nb_episodes + 1):
        
        state, _ = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0

        for t in range(nb_steps_per_episode):

            # Select and perform an action
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the transition in memory
            next_state = None if done else next_state
            memory.push(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if terminated:
                break

            # Training step
            if t > 1 and t % train_frequency == 0 and len(memory) >= batch_size:
                # print(f"Training at episode {episode}, step {t}")
                batch = memory.sample(batch_size)

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=agent.device,
                    dtype=torch.bool,
                )

                # Build batches and normalize pixel values to [0,1]
                state_batch = torch.FloatTensor(np.array(batch.state)).to(device)

                next_states_list = [s for s in batch.next_state if s is not None]
                if len(next_states_list) > 0:
                    non_final_next_states = torch.FloatTensor(np.array(next_states_list)).to(device)
                else:
                    non_final_next_states = torch.empty((0,), device=device)

                action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)

                # Compute Q(s_t, a)
                q_values = q_network(state_batch)
                state_action_values = q_values.gather(1, action_batch)

                # Compute Target and Loss
                q_next = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    # Only call target_network if we have non-final next states
                    if non_final_next_states.numel() != 0:
                        q_next[non_final_mask] = target_network(non_final_next_states).max(1)[0]
                expected_state_action_values = reward_batch + (gamma * q_next)

                loss = criterion(state_action_values.squeeze(), expected_state_action_values)

                if t % 100 == 0:
                    writer.add_scalar("Q-Values/Mean", q_values.mean().item(), episode * config.nb_steps_per_episode + t)
                    writer.add_scalar("Q-Values/Std", q_values.std().item(), episode * config.nb_steps_per_episode + t)
                    writer.add_scalar("Rewards/Positive", (reward_batch == 1).sum().item(), episode * config.nb_steps_per_episode + t)
                    print(f"Épisode {episode}, Step {t}: Q={q_values.mean().item():.1f}±{q_values.std().item():.1f}, "f"Loss={loss.item():.3f}, +1s={(reward_batch==1).sum().item()}")
             
                # Update Q-Network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loss_count += 1
            
            # Update target network (skip t==0 to avoid redundant immediate copy)
            if t > 0 and t % target_update_frequency == 0:
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
