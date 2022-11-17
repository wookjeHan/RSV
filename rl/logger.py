import os
from os.path import join as opj
from torch import save
from torch import Tensor
from collections import OrderedDict
from statistics import mean, stdev

class Logger:
    def __init__(self, base_dir, text_output_path, tab_output_path, snapshot_save_path):
        text_output_dir = opj(base_dir, text_output_path)
        tab_output_dir = opj(base_dir, tab_output_path)
        snapshot_save_dir = opj(base_dir, snapshot_save_path)

        os.makedirs(text_output_dir, exist_ok=True)
        os.makedirs(tab_output_dir, exist_ok=True)
        os.makedirs(snapshot_save_dir, exist_ok=True)

        text_output_path = opj(text_output_dir, "log.txt")
        tab_output_path = opj(tab_output_dir, "stat.csv")

        self.text_printer = open(text_output_path, 'w')
        self.tab_printer = open(tab_output_path, 'w')
        self.snapshot_save_dir = snapshot_save_dir

        print(f"Log text output as txt format to {text_output_path}")
        print(f"Log tabular output as csv format to {tab_output_path}")
        print(f"Snapshots will be saved to {snapshot_save_dir}")

        self.tab_title_printed = False
        self.buffer = OrderedDict()
        self.added = {}

    def add_scalar(self, key, value):
        assert key not in self.added
        if isinstance(value, Tensor):
            value = value.item()
        self.buffer[key] = value
        self.added[key] = 'scalar'

    def add_array_stat(self, key, tensor):
        assert isinstance(tensor, Tensor)
        assert len(tensor.shape) == 1

        values = tensor.tolist()
        if key in self.added:
            self.buffer[key] += values
        else:
            self.buffer[key] = values
            self.added[key] = 'array'

    def flush(self):
        text_output = ""
        methods = ['Min', 'Max', 'Mean', 'Stdev']
        method_dict= {
            'Min': min,
            'Max': max,
            'Mean': mean,
            'Stdev': stdev,
        }
        tab_values = []

        if not self.tab_title_printed:
            for key, value_type in self.added.items():
                if value_type == 'scalar':
                    self.tab_printer.write(key + ',')

                elif value_type == 'array':
                    for method in methods:
                        self.tab_printer.write(f"{key}/{method},")

            self.tab_printer.write('\n')
            self.tab_title_printed = True

        for key, value_type in self.added.items():
            if value_type == 'scalar':
                value = self.buffer[key]
                if isinstance(value, float):
                    value = round(value, 2)

                text_output += "{:<20s}|{:>10s}\n".format(key, str(value))
                tab_values.append(str(value))

            elif value_type == 'array':
                values = self.buffer[key]
                for method in methods:
                    value = method_dict[method](values)
                    if isinstance(value, float):
                        value = round(value, 2)

                    text_output += "{:<20s}|{:>10s}\n".format(f"{key}/{method}", str(value), 2)
                    tab_values.append(str(value))
                text_output += "------------------------------\n"
        text_output += "==============================\n"

        self.text_printer.write(text_output)
        self.tab_printer.write(",".join(tab_values) + "\n")
        self.text_printer.flush()
        self.tab_printer.flush()
        print(text_output, end='')

        self.buffer = OrderedDict()
        self.added = {}

    def save_snapshot(self, object, epoch):
        snapshot_save_path = opj(self.snapshot_save_dir, f"itr_{epoch}.pt")
        save(object, snapshot_save_path)
