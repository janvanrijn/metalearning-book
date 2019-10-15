import arff
import argparse
import fanova.fanova
import fanova.visualizer
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import openml
import openmlcontrib
import os
import sklearnbot


# to plot: <openml_pimp_root>/examples/plot/plot_fanova_aggregates.py
def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../sklearn-bot/data/svc.arff', type=str)
    parser.add_argument('--task_ids', type=int, nargs='+', default=[6, 58, 41, 28])
    parser.add_argument('--task_id_column', default='task_id', type=str)
    parser.add_argument('--classifier', default='svc', type=str)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/metabook')
    parser.add_argument('--output_format', type=str, default='pdf')
    parser.add_argument('--n_trees', default=16, type=int)
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--hyperparameter_name', type=str, default='svc__gamma')
    parser.add_argument('--exclude', type=str, nargs='+', default=['svc__max_iter', 'svc__degree', 'svc__coef0'])
    parser.add_argument('--resolution', default=100, type=int)
    args_, misc = parser.parse_known_args()

    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.info('Start %s: %s' % (os.path.basename(__file__), vars(args)))

    with open(args.dataset_path, 'r') as fp:
        arff_dataset = arff.load(fp)
    config_space_wrapper = sklearnbot.config_spaces.get_config_space(args.classifier, None)
    config_space_wrapper.wrap_in_fixed_pipeline()
    for exlude_param in args.exclude:
        config_space_wrapper.exclude_hyperparameter(exlude_param)
        config_space_wrapper.reset_conditions()
    config_space = config_space_wrapper.assemble()

    data = openmlcontrib.meta.arff_to_dataframe(arff_dataset, config_space)
    data = openmlcontrib.meta.integer_encode_dataframe(data, config_space)
    for exlude_param in args.exclude:
        del data[exlude_param]

    if args.scoring not in data.columns.values:
        raise ValueError('Could not find measure in dataset: %s' % args.measure)
    task_ids = data[args.task_id_column].unique()
    if args.task_ids:
        task_ids = args.task_ids

    os.makedirs(args.output_directory, exist_ok=True)
    output_file_gfx = os.path.join(args.output_directory, 'marginals_parameter_%s.%s'
                                   % ('_'.join(str(tid) for tid in sorted(args.task_ids)), args.output_format))
    hyperparameter_idx = config_space.get_idx_by_hyperparameter_name(args.hyperparameter_name)

    for t_idx, task_id in enumerate(task_ids):
        task = openml.tasks.get_task(task_id)
        data_name = task.get_dataset().name
        data_qualities = task.get_dataset().qualities

        logging.info('Running fanova on task %s (%d/%d)' % (task_id, t_idx + 1, len(task_ids)))
        data_task = data[data[args.task_id_column] == task_id]
        assert len(data_task) > 0
        del data_task[args.task_id_column]
        # now dataset is gone, and all categoricals are converted, we can convert to float
        data_task = data_task.astype(np.float)
        X_data = data_task[config_space.get_hyperparameter_names()].values
        y_data = data_task[args.scoring].values

        evaluator = fanova.fanova.fANOVA(X=X_data, Y=y_data, config_space=config_space, n_trees=args.n_trees)
        visualizer = fanova.visualizer.Visualizer(evaluator, config_space, '/tmp/', y_label=args.scoring)
        mean, std, grid = visualizer.generate_marginal(hyperparameter_idx, args.resolution)
        task_name = '%s (%d)' % (data_name, data_qualities['NumberOfFeatures'])

        plt.plot(grid, mean, label=task_name)
    plt.xscale('log')
    plt.xlabel('Hyperparameter value')
    plt.ylabel(args.scoring.replace('_', ' '))
    plt.legend()
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.savefig(output_file_gfx)
    logging.info('saved marginal plot to: %s' % output_file_gfx)


if __name__ == '__main__':
    run(read_cmd())
