import argparse
import logging
import matplotlib.pyplot as plt
import openml
import os


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
    setup_flowname = evaluations.set_index('setup_id')['flow_name'].to_dict()
    evaluations = evaluations[['task_id', 'setup_id', 'value']]

    # remove duplicates
    evaluations = evaluations.set_index(['task_id', 'setup_id'])
    evaluations = evaluations.loc[~evaluations.index.duplicated(keep='first')]
    evaluations = evaluations.reset_index()
    logging.info('Got %d run evaluations' % len(evaluations))

    # remove rows with missing results
    evaluations = evaluations.pivot(index='task_id', columns='setup_id', values='value').dropna()
    logging.info('Results are complete on %d tasks' % len(evaluations))

    fig_diagplot, ax_diagplot = plt.subplots()
    ax_diagplot.grid(linestyle='--')
    ax_diagplot.plot([0, 1], ls="-", color="gray")
    ax_diagplot.plot([0.2, 1.2], ls="--", color="gray")
    ax_diagplot.plot([-0.2, 0.8], ls="--", color="gray")
    ax_diagplot.scatter(evaluations[setup_ids[0]], evaluations[setup_ids[1]])
    ax_diagplot.set_xlabel('%s (vanilla)' % args.scoring)
    ax_diagplot.set_ylabel('%s (optimized)' % args.scoring)

    classifier_name = setup_flowname[args.setup_vanilla].split('(')[0]
    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory, 'effect_hpo_%s.%s' % (classifier_name, args.output_format))
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('Saved plot to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
