import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import matplotlib.pyplot as plt

# Base directory
BASE_DIR = "./data"

# List of directories for DQN and Double DQN
dqn_dirs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if "q1_dqn" in d]
double_dqn_dirs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if "q1_doubledqn" in d]

def extract_data_from_event_file(dir_path, tag):
    event_file = [f for f in os.listdir(dir_path) if "events.out.tfevents" in f][0]
    event_acc = EventAccumulator(os.path.join(dir_path, event_file))
    event_acc.Reload()
    _, step_nums, vals = zip(*event_acc.Scalars(tag))
    return step_nums, vals

def average_and_std_data_from_dirs(dirs, tag):
    all_vals = []
    max_length = 0
    
    # Extract data and find the maximum length
    for dir_path in dirs:
        _, vals = extract_data_from_event_file(dir_path, tag)
        all_vals.append(list(vals))  # Convert tuple to list
        if len(vals) > max_length:
            max_length = len(vals)
    
    # Pad shorter arrays
    for i in range(len(all_vals)):
        while len(all_vals[i]) < max_length:
            all_vals[i].append(all_vals[i][-1])
    
    avg_vals = np.mean(all_vals, axis=0)
    std_vals = np.std(all_vals, axis=0)
    return avg_vals, std_vals

def plot_with_error_bars(ax, avg_data, std_data, labels, title):
    for avg, std, label in zip(avg_data, std_data, labels):
        x = range(len(avg))  # Ensure x matches the length of avg
        ax.errorbar(x, avg, yerr=std, label=label)
    ax.set_title(title)
    ax.set_xlabel('Iterations / 1000')
    ax.set_ylabel('Returns')
    ax.legend()



def write_to_tensorboard(avg_data, log_dir, tag):
    writer = SummaryWriter(log_dir=log_dir)
    for i, val in enumerate(avg_data):
        writer.add_scalar(tag, val, i)
    writer.close()

if __name__ == "__main__":
    # for tag in ["Train_AverageReturn", "Train_BestReturn"]:
    #     dqn_avg = average_data_from_dirs(dqn_dirs, tag)
    #     double_dqn_avg = average_data_from_dirs(double_dqn_dirs, tag)
        
    #     write_to_tensorboard(dqn_avg, os.path.join(BASE_DIR, f"dqn_avg_{tag}"), tag)
    #     write_to_tensorboard(double_dqn_avg, os.path.join(BASE_DIR, f"double_dqn_avg_{tag}"), tag)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    tags = ["Train_AverageReturn", "Train_BestReturn"]
    
    # DQN data
    dqn_avg_data = []
    dqn_std_data = []
    for tag in tags:
        avg, std = average_and_std_data_from_dirs(dqn_dirs, tag)
        dqn_avg_data.append(avg)
        dqn_std_data.append(std)
    
    # Double DQN data
    double_dqn_avg_data = []
    double_dqn_std_data = []
    for tag in tags:
        avg, std = average_and_std_data_from_dirs(double_dqn_dirs, tag)
        double_dqn_avg_data.append(avg)
        double_dqn_std_data.append(std)
    
    # Plot for DQN
    plot_with_error_bars(axs[0], [dqn_avg_data[0], double_dqn_avg_data[0]], 
                         [dqn_std_data[0], double_dqn_std_data[0]], 
                         ["DQN", "Double DQN"], tags[0])
    
    # Plot for Double DQN
    plot_with_error_bars(axs[1], [dqn_avg_data[1], double_dqn_avg_data[1]], 
                         [dqn_std_data[1], double_dqn_std_data[1]], 
                         ["DQN", "Double DQN"], tags[1])
    
    plt.tight_layout()
    plt.savefig('q1_deliverable.png')
    plt.show()