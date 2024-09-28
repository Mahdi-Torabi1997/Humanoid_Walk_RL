import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import imageio
from IPython.display import Video

# Function to save the video
def save_video(frames, filename="output_video.mp4", fps=60):  # Double the FPS for 2x speed
    """Save a list of frames as a video."""
    with imageio.get_writer(filename, fps=fps) as video:
        for frame in frames:
            video.append_data(frame)

# Test the model and record the video
def test_and_record_video(env, sb3_algo, path_to_model, video_filename="output_video.mp4", max_duration_seconds=60, speedup_factor=2):
    """Test the model and save the video of the bot's actions, speed it up, and limit duration."""
    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return

    obs, _ = env.reset()
    done = False
    frames = []  # List to store frames for the video
    
    frame_rate = 30  # Original frame rate
    total_frames = int(max_duration_seconds * frame_rate)  # Total frames for 1 minute
    collected_frames = 0
    
    while collected_frames < total_frames:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        
        # Capture frame from the environment render
        frame = env.render()  
        frames.append(frame)
        
        collected_frames += 1
        
        if done:
            obs, _ = env.reset()

    # Save the video at 2x speed (which effectively halves the duration)
    save_video(frames, video_filename, fps=frame_rate * speedup_factor)

    # Return the filename for the video
    return video_filename

# Load the environment with rendering enabled
gymenv_name = 'Humanoid-v4'  # Replace this with your desired Gym environment
sb3_algo = 'SAC'  # Choose the algorithm you're using ('SAC', 'TD3', or 'A2C')
test_model_path = 'models/SAC_1025000.zip'  # Replace with the path to the specific model file

# Load the environment with 'rgb_array' rendering
gymenv = gym.make(gymenv_name, render_mode='rgb_array')

# Run the test and record the video, setting it to show only the first minute at 2x speed
video_filename = test_and_record_video(gymenv, sb3_algo, test_model_path, max_duration_seconds=60, speedup_factor=2)

# Display the video inside the notebook
Video(video_filename)
