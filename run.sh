#!/usr/bin/env bash

# Run all of the experiments.

function main() {
  trap exit SIGINT

  # Setup online psl examples
  # Downloads data and PSL jar.
  ./scripts/setup_online_psl_examples.sh

  # Run atom update experiments, i.e, Movielens-1m and Bikeshare.
  ./scripts/run_atom_update_experiments.sh

  # Run template modification experiments, i.e, Epinions.
  ./scripts/run_template_modification_experiments.sh
}

main "$@"