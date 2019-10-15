import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=6)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/metabook')
    parser.add_argument('--output_format', type=str, default='pdf')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--n_flows', type=int, default=30)
    args_ = parser.parse_args()
    return args_


def process_name(name: str):
    splitted = name.split('.')
    if splitted[0] == 'weka':
        brackets = name.count('_')
        return name.replace('_', '(') + (')' * brackets)
    if splitted[0] == 'sklearn':
        return 'sklearn.Pipeline(..., %s' %splitted[-1]
    logging.info('undetermined flow: %s' % name)
    if len(name) > 40:
        return '[...]%s' % name[-20:]
    return name


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    evaluations = openml.evaluations.list_evaluations(
        args.scoring, 0, 1000, task=[args.task_id], output_format='dataframe')
    flows = openml.flows.list_flows(output_format='dataframe').set_index('id')

    evaluations = evaluations.join(other=flows, on='flow_id', lsuffix='e')
    evaluations = evaluations.groupby('name')['value'].agg([max, len, pd.Series.tolist])
    evaluations = evaluations.sort_values(['max'], ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    scatter_name = list()
    scatter_x = list()
    scatter_y = list()
    for idx, (name, row) in enumerate(evaluations.iterrows()):
        if idx >= args.n_flows:
            continue
        scatter_name.append(process_name(name))
        for value in row['tolist']:
            scatter_x.append(idx)
            scatter_y.append(value)

    ax.scatter(scatter_x, scatter_y)
    ax.xaxis.set_ticks(np.arange(0, args.n_flows, 1))
    ax.set_xticklabels(scatter_name, rotation=45, ha='right')
    ax.set_xlabel(None)
    ax.set_ylabel(args.scoring.replace('_', ' '))

    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory, 'comparing_flows_task_%d.%s' % (args.task_id,
                                                                                      args.output_format))
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('Saved plot to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
