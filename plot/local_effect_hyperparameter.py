import argparse
import logging
import matplotlib.pyplot as plt
import openml
import os
import pandas as pd
import sklearn
import sklearnbot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_ids', type=int, nargs='+', default=[3547, 6, 58, 41, 28])
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/metabook')
    parser.add_argument('--output_format', type=str, default='pdf')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--classifier', type=str, default='svc')
    parser.add_argument('--hyperparameter_name', type=str, default='svc__gamma')
    parser.add_argument('--hyperparameter_values', type=float, nargs='+', default=[2 ** i for i in range(-5, 7)])
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    config_space_wrapper = sklearnbot.config_spaces.get_config_space(args.classifier, args.random_seed)
    config_space_wrapper.wrap_in_fixed_pipeline()
    config_space = config_space_wrapper.assemble()

    setup_ids = []
    task_names = {}
    # this generates all runs and collects all setup ids
    for task_id in args.task_ids:
        task = openml.tasks.get_task(task_id)
        data_name = task.get_dataset().name
        data_qualities = task.get_dataset().qualities
        data_tuple = (task.task_id, data_name, data_qualities['NumberOfFeatures'], data_qualities['NumberOfInstances'])
        logging.info('Obtained task %d (%s); %s attributes; %s observations' % data_tuple)
        task_names[task_id] = '%s (%d)' % (data_name, data_qualities['NumberOfFeatures'])

        # obtain deserialized classifier
        nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
        numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])

        classifier = sklearnbot.sklearn.as_pipeline(config_space, nominal_indices, numeric_indices)

        for value in args.hyperparameter_values:
            logging.info('Task %d, %s=%f' % (task_id, args.hyperparameter_name, value))
            classifier.set_params(**{args.hyperparameter_name: value})
            flow = openml.extensions.get_extension_by_model(classifier).model_to_flow(classifier)
            flow_id = openml.flows.flow_exists(flow.name, flow.external_version)
            # for updating flow id
            flow = openml.flows.get_flow(flow_id)
            flow.model = classifier
            setup_id = openml.setups.setup_exists(flow)
            if setup_id is False:
                logging.info('=== Start running flow ===')
                run = openml.runs.run_flow_on_task(flow, task)
                run = run.publish()
                run = openml.runs.get_run(run.run_id)  # for updating setup id
                setup_id = run.setup_id
                score = run.get_metric_fn(sklearn.metrics.accuracy_score)
                logging.info('Task %d - %s; Accuracy: %0.2f; Run id: %d' % (task_id,
                                                                            task.get_dataset().name,
                                                                            score.mean(),
                                                                            run.run_id))
            setup_ids.append(setup_id)

    evaluations = openml.evaluations.list_evaluations_setups(args.scoring, task=args.task_ids, setup=setup_ids)
    subflow = flow.get_subflow(args.hyperparameter_name.split('__')[:-1])
    openml_param_name = '%s(%s)_%s' % (subflow.name, subflow.version, args.hyperparameter_name.split('__')[-1])
    result_view = pd.DataFrame([{
        'hyperparameter_value': evaluation['parameters'][openml_param_name],
        'value': evaluation['value'],
        'task': task_names[evaluation['task_id']]
    } for _, evaluation in evaluations.iterrows()])

    print(result_view)


if __name__ == '__main__':
    run(parse_args())
