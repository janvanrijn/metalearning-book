import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import openml
import os
import pandas as pd
import sklearnbot
import typing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=6)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/metabook')
    parser.add_argument('--output_format', type=str, default='pdf')
    parser.add_argument('--setup_vanilla', type=int, default=8254067)
    parser.add_argument('--setup_hpo', type=int, default=8254068)
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--study_id', type=int, default=99)
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    setup_ids = [args.setup_vanilla, args.setup_hpo]

    tasks = openml.study.get_suite(args.study_id).tasks

    # Downloads all evaluation records related to this study
    evaluations = openml.evaluations.list_evaluations(
        args.scoring, setup=setup_ids, task=tasks, output_format='dataframe')
    evaluations = evaluations[['task_id', 'setup_id', 'value']]

    # remove duplicates
    evaluations = evaluations.set_index(['task_id', 'setup_id'])
    evaluations = evaluations.loc[~evaluations.index.duplicated(keep='first')]
    evaluations = evaluations.reset_index()

    # remove rows with missing results
    evaluations = evaluations.pivot(index='task_id', columns='setup_id', values='value').dropna()

    fig_diagplot, ax_diagplot = plt.subplots()
    ax_diagplot.grid(linestyle='--')
    ax_diagplot.plot([0, 1], ls="-", color="gray")
    ax_diagplot.plot([0.2, 1.2], ls="--", color="gray")
    ax_diagplot.plot([-0.2, 0.8], ls="--", color="gray")
    ax_diagplot.scatter(evaluations[setup_ids[0]], evaluations[setup_ids[1]])
    ax_diagplot.set_xlabel('%s %s' % (args.scoring, setup_ids[0]))
    ax_diagplot.set_ylabel('%s %s' % (args.scoring, setup_ids[1]))
    plt.show()


if __name__ == '__main__':
    run(parse_args())
