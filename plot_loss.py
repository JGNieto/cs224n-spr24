from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))

MODAL_FILEPATH = os.path.join(current_directory, 'modal_results')

def find_first_csv(directory) -> Union[str, None]:
    try:
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                return os.path.join(directory, file)
    except FileNotFoundError:
        return None
    return None

def plot_data(plt, data, task, train_color, dev_color, args):
    # Divide by max to normalize
    dev_max = data[f'{task}_dev_loss'].max()
    train_max = data[f'{task}_train_loss'].max()
    overall_max = max(dev_max, train_max)

    data[f'{task}_dev_loss'] = data[f'{task}_dev_loss'] / overall_max
    data[f'{task}_train_loss'] = data[f'{task}_train_loss'] / overall_max

    # Smooth data:
    if args.smooth:
        data[f'{task}_dev_loss'] = data[f'{task}_dev_loss'].rolling(window=5).mean()
        data[f'{task}_train_loss'] = data[f'{task}_train_loss'].rolling(window=5).mean()

    plt.plot(data['iter_adj'], data[f'{task}_dev_loss'], label=f'{task.capitalize()} Dev Loss', color=dev_color)
    plt.plot(data['iter_adj'], data[f'{task}_train_loss'], label=f'{task.capitalize()} Train Loss', color=train_color)

def run(args):
    exp_name = args.experiment_name
    print(f"Plotting {exp_name} with {', '.join(args.plot)}...")

    dir_path = os.path.join(MODAL_FILEPATH, exp_name, 'logs')
    file_path = find_first_csv(dir_path)

    if file_path is None:
        print(f"No CSV file found in {dir_path}")
        return

    data = pd.read_csv(file_path)

    # Compute the total dev and train losses
    data['total_dev_loss'] = data[['sst_dev_loss', 'para_dev_loss', 'sts_dev_loss']].sum(axis=1)
    data['total_train_loss'] = data[['sst_train_loss', 'para_train_loss', 'sts_train_loss']].sum(axis=1)

    max_iter = data['iter'].max()

    data["iter_adj"] = data["epoch"] + (data["iter"] / max_iter)

    # Plotting
    plt.figure(figsize=(6, 6))

    # Plot dev and train losses
    for plot in args.plot:
        if plot == 'total':
            plot_data(plt, data, 'total', 'blue', 'red', args)
        elif plot == 'sts':
            plot_data(plt, data, 'sts', 'green', 'orange', args)
        elif plot == 'sst':
            plot_data(plt, data, 'sst', 'purple', 'pink', args)
        elif plot == 'para':
            plot_data(plt, data, 'para', 'brown', 'olive', args)

    # Labeling the axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Set the x-ticks to show epoch numbers only
    epochs = data['epoch'].unique()
    epoch_ticks = [data[data['epoch'] == epoch]['iter_adj'].iloc[0] for epoch in epochs]
    plt.xticks(epoch_ticks, epochs)

    # Title and legend
    plt.title('Training and Development Loss Over Iterations')
    plt.legend()

    plt.savefig(f"plots/{exp_name}.png")
    plt.savefig(f"plots/{exp_name}.pdf")

    # Show the plot
    if args.show:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('experiment_name', type=str, help='Name of the experiment (letters only)')
    parser.add_argument('--plot', nargs='+', choices=['total', 'sts', 'sst', 'para', 'all'],
                        help='List of plots to generate (e.g. --plot total sts sst)')
    parser.add_argument('--smooth', action='store_true', help='Smooth the data')
    parser.add_argument('--axis_limits', nargs=2, type=int, help='Set the limits of the x-axis in epochs (e.g. --axis_limits 0 10)')
    parser.add_argument('--show', '-s', action='store_true', help='Show the plot')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.plot is None:
        args.plot = ['total']
    elif 'all' in args.plot:
        args.plot = ['total', 'sts', 'sst', 'para']

    run(args)

