import json
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_log(logdir: str):
    event_acc = EventAccumulator(logdir).Reload()
    scalar_values_extractor = lambda name: [val.value for val in event_acc.Scalars(name)]
    scalar_tags = event_acc.Tags()['scalars']
    scalar_dict = {tag: scalar_values_extractor(tag) for tag in scalar_tags}
    return scalar_dict

def read_hparams(logdir: str):
    with open(os.path.join(logdir, 'hparams.json'), 'r') as hparams_file:
        hparams = json.load(hparams_file)
    return hparams

def dump_dict_to_file(dict: dict, filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, "w") as tensorboard_file:
            json.dump(dict, tensorboard_file)