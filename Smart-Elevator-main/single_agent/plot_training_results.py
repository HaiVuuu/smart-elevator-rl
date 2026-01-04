import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def extract_rewards_from_tfevents(log_dir, reward_tag='rollout/ep_rew_mean'):
    """Extracts steps and reward values from a TensorBoard log directory."""
    
    event_file = None
    for f in os.listdir(log_dir):
        if "events.out.tfevents" in f:
            event_file = os.path.join(log_dir, f)
            break
            
    if event_file is None:
        print(f"Warning: No tfevents file found in {log_dir}")
        return [], []

    steps = []
    rewards = []
    try:
        for summary in tf.compat.v1.train.summary_iterator(event_file):
            for value in summary.summary.value:
                if value.tag == reward_tag:
                    steps.append(summary.step)
                    rewards.append(value.simple_value)
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        
    return steps, rewards

def main():
    """Main function to generate and save the plot."""
    
    log_dirs = {
        "LR=0.0001": os.path.join(".", "logs", "DQN_1_1"),
        "LR=0.0005": os.path.join(".", "logs", "DQN_1_3"),
        "LR=0.001": os.path.join(".", "logs", "DQN_1_4")
    }
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#f0f0f0')  # Set a light grey background

    print("Processing log files...")
    for label, log_dir in log_dirs.items():
        if not os.path.isdir(log_dir):
            print(f"Directory not found: {log_dir}")
            continue
            
        steps, rewards = extract_rewards_from_tfevents(log_dir)
        if steps:
            # Smooth the curve for better visualization using a moving average
            rewards_smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')
            steps_smooth = steps[:len(rewards_smooth)]
            ax.plot(steps_smooth, rewards_smooth, label=label, alpha=0.8)
            print(f"Plotted {label} with {len(steps)} data points.")

    ax.set_title('DQN Training Performance by Learning Rate', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Mean Episode Reward ', fontsize=12)
    ax.legend(title='Learning Rate', fontsize=10)
    
    # Format x-axis to show full integer numbers every 500k steps
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250000))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))

    # Save the plot to a file
    output_filename = 'dqn_training_comparison.png'
    plt.savefig(output_filename, bbox_inches='tight')
    
    print(f"\nPlot successfully saved as '{output_filename}'")
    

if __name__ == '__main__':
    main()
