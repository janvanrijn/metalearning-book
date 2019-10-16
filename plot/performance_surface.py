import argparse
import openml
import matplotlib.pyplot as plt
import json
import logging
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=6)
    parser.add_argument('--flow_id', type=int, default=8353)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/metabook')
    parser.add_argument('--output_format', type=str, default='pdf')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    evals = openml.evaluations.list_evaluations_setups(args.scoring,
                                                       flow=[args.flow_id],
                                                       task=[args.task_id],
                                                       output_format='dataframe')
    setups_evals = pd.DataFrame([dict(**{name: json.loads(value) for name, value in setup['parameters'].items()},
                                      **{'value': setup['value']})
                                 for _, setup in evals.iterrows()])

    setups_evals = setups_evals[setups_evals['sklearn.svm.classes.SVC(16)_kernel'] == 'rbf']

    C = [float(x) for x in setups_evals['sklearn.svm.classes.SVC(16)_C']]
    gamma = [float(x) for x in setups_evals['sklearn.svm.classes.SVC(16)_gamma']]
    score = [float(x) for x in setups_evals['value']]

    fig, ax = plt.subplots()
    ax.plot(C, gamma, 'ko', ms=1)
    cntr = ax.tricontourf(C, gamma, score, levels=12, cmap="RdBu_r")
    fig.colorbar(cntr, ax=ax, label=args.scoring.replace('_', ' '))
    ax.set(xlim=[min(C), max(C)], ylim=[min(gamma), max(gamma)], xlabel="C", ylabel="gamma")

    ax.set_xscale('log')
    ax.set_yscale('log')

    output_file_gfx = os.path.join(args.output_directory, 'performance_surface_%d.%s'
                                   % (args.task_id, args.output_format))
    plt.savefig(output_file_gfx)
    logging.info('saved marginal plot to: %s' % output_file_gfx)


if __name__ == '__main__':
    run(parse_args())
