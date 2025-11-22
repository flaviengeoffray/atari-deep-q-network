import torch
import yaml
import click

from dqn.env import init_env
from dqn.agent import DQNAgent
from dqn.dqn import DQN
from dqn.train import TrainingConfig, training, evaluate


def read_config(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)["training_config"]

    print("Training Configuration:")
    for k, v in config_dict.items():
        print(f"{k}: {v}")
    return TrainingConfig(**config_dict)


def load_checkpoint(model: DQN, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


@click.command()
@click.option('--train/--no-train', default=False, help='Run training loop.')
@click.option('--eval/--no-eval', default=False, help='Run evaluation.')
@click.option('--render-human/--no-render-human', default=False, help='Render evaluation in human mode.')
@click.option('--create-video/--no-create-video', default=False, help='Create a video of the evaluation.')
@click.option('--config', 'config_path', default='training_config.yml', type=click.Path(), help='Path to training config YAML.')
@click.option('--checkpoint', 'checkpoint_path', default='best_run/dqn_pong_5000.pth', type=click.Path(), help='Path to model checkpoint for evaluation.')
@click.option('--env', 'env_id', default='ALE/Pong-v5', help='Gym environment id to use.')
@click.option('--episodes', default=10, type=int, help='Number of episodes for evaluation.')
def cli(train, eval, render_human, create_video, config_path, checkpoint_path, env_id, episodes):
    """
    CLI to train and/or evaluate the DQN agent.
    """
    if train:
        cfg = read_config(config_path)
        training_env = init_env(env_id, render_mode="rgb_array")
        training(training_env, cfg)

    if eval:
        render_type = "human" if render_human else "rgb_array"
        if render_human:
            print("Rendering evaluation in human mode.")
            print("Note: Video recording is disabled in human render mode.")
            create_video = False

        env = init_env(env_id, render_mode=render_type)
        agent = DQNAgent(
            DQN(env.action_space.n),
            env.action_space.n,
            epsilon_start=0.0,
            epsilon_decay=1,
            epsilon_end=0.0,
            device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        )

        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            load_checkpoint(agent.model, checkpoint_path)

        print(f"Evaluating on {env_id} for {episodes} episodes (create_video={create_video}, render_human={render_human})")
        avg_reward = evaluate(env, agent, n_episodes=episodes, create_video=create_video)
        print(f"Average Reward over {episodes} episodes: {avg_reward}")
    
    if train is False and eval is False:
        print("Please specify at least one of --train or --eval options.")
        print("Use --help for more information.")

if __name__ == "__main__":
    cli()
