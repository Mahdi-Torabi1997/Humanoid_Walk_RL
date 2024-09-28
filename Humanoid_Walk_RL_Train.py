import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, sb3_algo):
    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

def test(env, sb3_algo, path_to_model):
    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break

# In Jupyter, define the environment and algorithm manually
gymenv_name = 'Humanoid-v4'  # Replace this with your desired Gym environment
sb3_algo = 'SAC'  # Choose 'SAC', 'TD3', or 'A2C'
train_mode = True  # Set to True to train, False to test
test_model_path = 'models/SAC_25000.zip'  # Provide the path to your saved model for testing

# Running the training or testing process
if train_mode:
    gymenv = gym.make(gymenv_name, render_mode=None)
    train(gymenv, sb3_algo)
else:
    if os.path.isfile(test_model_path):
        gymenv = gym.make(gymenv_name, render_mode='human')
        test(gymenv, sb3_algo, path_to_model=test_model_path)
    else:
        print(f'{test_model_path} not found.')
