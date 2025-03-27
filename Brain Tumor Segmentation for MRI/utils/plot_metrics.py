from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

def extract_metrics(logdir):
    acc = EventAccumulator(logdir)
    acc.Reload()

    data = {}
    for tag in acc.Tags()['scalars']:
        data[tag] = [(x.step, x.value) for x in acc.Scalars(tag)]
    return data

def plot_metrics(metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    for metric, values in metrics.items():
        steps, vals = zip(*values)
        plt.plot(steps, vals, label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    log_dir = "runs/experiment_1"
    metrics = extract_metrics(log_dir)
    plot_metrics(metrics)
