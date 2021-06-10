import os
import sys

import pandas as pd
import numpy as np


def main(inferred_predicates_path, split_targets_path):
    inferred_predicates_df = pd.read_csv(inferred_predicates_path, header=None, sep="\t")
    inferred_predicates_df = inferred_predicates_df.set_index(list(range(inferred_predicates_df.shape[1] - 1)))

    split_targets_df = pd.read_csv(split_targets_path, header=None, sep="\t")
    split_targets_df = split_targets_df.set_index(list(range(split_targets_df.shape[1])))

    hot_start_atom_df = inferred_predicates_df.reindex(split_targets_df.index)
    # Fill potentially missing values with random [0, 1) value.
    hot_start_atom_df[hot_start_atom_df.columns[0]] = hot_start_atom_df[hot_start_atom_df.columns[0]].apply(
        lambda x: np.random.random() if pd.isna(x) else x)

    # Write hotstart file.
    hot_start_atom_df.to_csv(os.path.join(os.path.dirname(split_targets_path), "hotstart_target.txt"),
                             sep="\t", header=False)


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 2 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s <inferred_predicates_path> <split_targets_path>" % (executable), file=sys.stderr)
        sys.exit(1)

    arg_1 = args.pop(0)
    arg_2 = args.pop(0)
    return arg_1, arg_2


if __name__ == '__main__':
    inferred_predicates_path, split_targets_path = _load_args(sys.argv)
    main(inferred_predicates_path, split_targets_path)
