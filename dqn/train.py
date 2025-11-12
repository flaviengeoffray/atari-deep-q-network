import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
import tqdm

from dqn.dqn import DQN
from dqn.agent import DQNAgent 
from dqn.memory import ReplayMemory, Transition


class TrainingConfig:
    nb_episodes: int = 1000
    nb_steps_per_episode: int = 10000
    memory_capacity: int = 100000
    batch_size: int = 32
    train_frequency: int = 4
    lr: float = 0.00025
    gamma: float = 0.99
    target_update_frequency: int = 1000
    checkpoint_frequency: int = 100

def training(env: gym.Env, config: TrainingConfig = TrainingConfig()):
    
    # Unpack config
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
    q_network = DQN(env.action_space.n)
    target_network = DQN(env.action_space.n)
    target_network.load_state_dict(q_network.state_dict())
    agent = DQNAgent(q_network, env.action_space.n, epsilon_start=1.0, epsilon_decay=500000, epsilon_end=0.1)
    memory = ReplayMemory(memory_capacity)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
    writer = SummaryWriter("runs/dqn_experiment")

    for episode in tqdm.tqdm(range(nb_episodes), desc="Training Episodes"):
        
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

            # Training step
            if len(memory) >= batch_size and t % train_frequency == 0:
                
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=agent.device,
                    dtype=torch.bool,
                )
                non_final_next_states = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(agent.device)
                state_batch = torch.FloatTensor(np.array(batch.state)).to(agent.device)
                action_batch = torch.tensor(batch.action, device=agent.device).unsqueeze(1)
                reward_batch = torch.FloatTensor(np.array(batch.reward)).to(agent.device)

                # Compute Loss
                state_action_values = q_network(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(batch_size, device=agent.device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]
                expected_state_action_values = reward_batch + (gamma * next_state_values)

                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values.squeeze(), expected_state_action_values)

                # Update Q-Network
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0) # Gradient clipping
                optimizer.step()

                total_loss += loss.item()
                loss_count += 1
            
            # Update target network
            if t % target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            if done:
                break
        
        # Logging
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            writer.add_scalar("Loss/Training", avg_loss, episode)
        writer.add_scalar("Reward/Training", total_reward, episode)
        writer.add_scalar("Exploration/ε", agent.epsilon, episode)

        # Save checkpoint
        if episode % checkpoint_frequency == 0:
            torch.save(q_network.state_dict(), f"checkpoints/dqn_pong_{episode}.pth")
    
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
