import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

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

def average_data_from_dirs(dirs, tag):
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
    return avg_vals


def write_to_tensorboard(avg_data, log_dir, tag):
    writer = SummaryWriter(log_dir=log_dir)
    for i, val in enumerate(avg_data):
        writer.add_scalar(tag, val, i)
    writer.close()

if __name__ == "__main__":
    for tag in ["Train_AverageReturn", "Train_BestReturn"]:
        dqn_avg = average_data_from_dirs(dqn_dirs, tag)
        double_dqn_avg = average_data_from_dirs(double_dqn_dirs, tag)
        
        write_to_tensorboard(dqn_avg, os.path.join(BASE_DIR, f"dqn_avg_{tag}"), tag)
        write_to_tensorboard(double_dqn_avg, os.path.join(BASE_DIR, f"double_dqn_avg_{tag}"), tag)
