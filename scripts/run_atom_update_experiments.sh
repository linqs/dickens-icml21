#!/usr/bin/env bash

# Run all the experiments.

readonly BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly EXAMPLE_DIR="${BASE_DIR}/../online-psl-examples"
readonly RESULTS_DIR="${BASE_DIR}/../results"

declare -A INFERENCE_METHOD_OPTIONS
INFERENCE_METHOD_OPTIONS[SGD]='--infer=SGDInference'
INFERENCE_METHOD_OPTIONS[ADMM]='--infer=ADMMInference'
INFERENCE_METHOD_OPTIONS[SGD_TI]='--infer=SGDStreamingInference'

declare -A EXAMPLE_OPTIONS
EXAMPLE_OPTIONS[bikeshare]='-D sgd.maxiterations=500 -D reasoner.tolerance=1e-6f '
EXAMPLE_OPTIONS[movielens-1m]='-D sgd.maxiterations=500 -D reasoner.tolerance=1e-5f '

declare -A VARIANT_OPTIONS
VARIANT_OPTIONS[bikeshare_time_series_NON_POWERSET]='-D sgd.learningrate=1.0 '
VARIANT_OPTIONS[bikeshare_time_series_POWERSET]='-D sgd.learningrate=1.0 '
VARIANT_OPTIONS[bikeshare_time_series_OFFLINE]='-D sgd.learningrate=1.0 '
VARIANT_OPTIONS[movielens-1m_online_NON_POWERSET]='-D sgd.learningrate=0.1 '
VARIANT_OPTIONS[movielens-1m_online_POWERSET]='-D sgd.learningrate=0.1 '
VARIANT_OPTIONS[movielens-1m_online_OFFLINE]='-D sgd.learningrate=0.1 '
VARIANT_OPTIONS[movielens-1m_time_series_NON_POWERSET]='-D sgd.learningrate=1.0 '
VARIANT_OPTIONS[movielens-1m_time_series_POWERSET]='-D sgd.learningrate=0.1 '
VARIANT_OPTIONS[movielens-1m_time_series_OFFLINE]='-D sgd.learningrate=0.1 '

declare -A EXAMPLE_TARGET_FILE
EXAMPLE_TARGET_FILE[bikeshare]='Demand_target.txt'
EXAMPLE_TARGET_FILE[movielens-1m]='rating_target.txt'

declare -A EXAMPLE_INFERRED_FILE
EXAMPLE_INFERRED_FILE[bikeshare]='DEMAND.txt'
EXAMPLE_INFERRED_FILE[movielens-1m]='RATING.txt'

declare -A EXAMPLE_VARIANT
EXAMPLE_VARIANT[bikeshare]='bikeshare_time_series'
EXAMPLE_VARIANT[movielens-1m]='movielens-1m_online movielens-1m_time_series'

declare -A INITIALIZATION_OPTIONS
INITIALIZATION_OPTIONS['ATOM']='-D inference.initialvalue=ATOM -D inference.onlinehotstart=true'
INITIALIZATION_OPTIONS['RANDOM']='-D inference.initialvalue=RANDOM -D inference.onlinehotstart=false'

readonly SUPPORTED_EXAMPLES='movielens-1m bikeshare'
readonly ATOM_INITIALIZATIONS='ATOM RANDOM'
readonly OFFLINE_INFERENCE_METHODS='SGD_TI'
readonly ONLINE_GROUNDING_METHODS='NON_POWERSET POWERSET'

function run_online() {
  local example_name=$1
  local variant=$2
  local intialization=$3
  local fold=$4
  local timing=$5

  # Declare paths to output files.
  local out_directory=""
  local out_path=""
  local err_path=""
  local experiment_options=""

  for grounding_method in ${ONLINE_GROUNDING_METHODS}; do
    echo "Running PSL ${example_name}-${variant} (fold:${fold} -- initialization:${initialization} -- grounding_method:${grounding_method})."

    # Declare paths to output files.
    if [[ $timing == "false" ]]; then
      out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/online/${grounding_method}/${intialization}"
    else
      out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/online_timing/${grounding_method}/${intialization}"
    fi
    out_path="${out_directory}/out.txt"
    err_path="${out_directory}/out.err"
    experiment_options="${EXAMPLE_OPTIONS[${example_name}]} ${VARIANT_OPTIONS[${variant}_${grounding_method}]} ${INITIALIZATION_OPTIONS[${initialization}]}"

    if [[ -e "${out_path}" ]]; then
      echo "Output file already exists, skipping: ${out_path}"
    else
      mkdir -p ${out_directory}
      pushd . > /dev/null
         cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

         # Set the variant.
         sed -i "s@data/${example_name}/${example_name}.*/[0-9]\+/eval/@data/${example_name}/${variant}/00/eval/@g" "${example_name}-eval.data"
         # Set the data fold.
         sed -i "s/data\/${example_name}\/${variant}\/[0-9]\+/data\/${example_name}\/${variant}\/${fold}/g" "${example_name}-eval.data"
         # Set the data split.
         sed -i "s/eval\/[0-9]\+/eval\/00/g" "${example_name}-eval.data"
         # Set the commands location in run_client.sh.
         sed -i "s@^readonly COMMAND_FILE=.*'\$@readonly COMMAND_FILE='../data/${example_name}/${variant}/${fold}/eval/commands.txt'@g" run_client.sh
         # Ensure a previously failed offline run didn't dirty the target entry of eval.data file.
         sed -i "s@hotstart_target.txt@${EXAMPLE_TARGET_FILE[${example_name}]}@g" "${example_name}-eval.data"
         # Set the logging level depending on whether this is timing experiment
         if [[ $timing == "false" ]]; then
          sed -i "s/^readonly ADDITIONAL_SERVER_OPTIONS='.*'$/readonly ADDITIONAL_SERVER_OPTIONS='-D log4j.threshold=TRACE'/g" run_server.sh
          sed -i "s/^readonly ADDITIONAL_CLIENT_OPTIONS='.*'$/readonly ADDITIONAL_CLIENT_OPTIONS='-D log4j.threshold=TRACE'/g" run_client.sh
         else
          sed -i "s/^readonly ADDITIONAL_SERVER_OPTIONS='.*'$/readonly ADDITIONAL_SERVER_OPTIONS=''/g" run_server.sh
          sed -i "s/^readonly ADDITIONAL_CLIENT_OPTIONS='.*'$/readonly ADDITIONAL_CLIENT_OPTIONS=''/g" run_client.sh
         fi

         # Add to options for run depending on grounding_method.
         if [[ $grounding_method == "POWERSET" ]]; then
           experiment_options="$experiment_options -D partialgrounding.powerset=true"
         fi

         ./run.sh ${experiment_options} > "${out_path}" 2> "${err_path}"

         mv inferred-predicates "${out_directory}/"
         mv serverResponses "${out_directory}/"
         cp "./${example_name}-eval.data" "${out_directory}/${example_name}-eval.data"
         cp "./${example_name}.psl" "${out_directory}/${example_name}.psl"
         cp "./run_client.sh" "${out_directory}/run_client.sh"
         cp "./run_server.sh" "${out_directory}/run_server.sh"
         cp "./out_server.txt" "${out_directory}/out_server.txt"
         cp "./out_server.err" "${out_directory}/out_server.err"
         cp "./out_client.txt" "${out_directory}/out_client.txt"
         cp "./out_client.err" "${out_directory}/out_client.err"
      popd > /dev/null
    fi
  done
}

function run_offline() {
  local example_name=$1
  local variant=$2
  local initialization=$3
  local fold=$4
  local timing=$5

  local out_path=""
  local err_path=""
  local out_directory=""
  local split_targets_dir=""
  local split_targets_path=""
  local prev_split=""

  for inference_method in ${OFFLINE_INFERENCE_METHODS}; do
    if [[ $timing == "false" ]]; then
      out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/offline/${inference_method}/${initialization}"
    else
      out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/offline_timing/${inference_method}/${initialization}"
    fi

    # Set the inference method in the run_offline script.
    pushd . > /dev/null
     cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

     if [[ $timing == "false" ]]; then
      sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='.*'$/readonly ADDITIONAL_EVAL_OPTIONS='${INFERENCE_METHOD_OPTIONS[${inference_method}]} -D log4j.threshold=TRACE'/g" run_offline.sh
     else
      sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='.*'$/readonly ADDITIONAL_EVAL_OPTIONS='${INFERENCE_METHOD_OPTIONS[${inference_method}]}'/g" run_offline.sh
     fi
    popd > /dev/null

    for split in $(ls -d ./online-psl-examples/${example_name}/data/${example_name}/${variant}/${fold}/eval/*/ | cut -f 9 -d '/'); do
      echo "Running PSL ${example_name}-${variant} (fold:${fold} -- initialization:${initialization} -- time_step:${split})."

      # Declare paths to output files.
      out_path="${out_directory}/${split}/out.txt"
      err_path="${out_directory}/${split}/out.err"
      # Declare paths to target files.
      split_targets_dir="../data/${example_name}/${variant}/${fold}/eval/${split}"
      split_targets_path="${split_targets_dir}/${EXAMPLE_TARGET_FILE[${example_name}]}"

      if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
      else
       mkdir -p "${out_directory}/${split}"
       [[ -d "${out_directory}/inferred-predicates" ]] || mkdir -p "${out_directory}/inferred-predicates"
       pushd . > /dev/null
         cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

         # Set the variant.
         sed -i "s@data/${example_name}/${example_name}.*/[0-9]\+/eval/@data/${example_name}/${variant}/00/eval/@g" "${example_name}-eval.data"
         # Set the data fold.
         sed -i "s/data\/${example_name}\/${variant}\/[0-9]\+/data\/${example_name}\/${variant}\/${fold}/g" "${example_name}-eval.data"
         # Set the data splits.
         sed -i "s/eval\/[0-9]\+/eval\/${split}/g" "${example_name}-eval.data"

         if [ $initialization == "ATOM" ] && [ $split != "00" ]; then
           # Join inferred predicates and split predicates to one file. Keep inferred predicates.
           join_experiment_results "${out_directory}/inferred-predicates/${prev_split}/${EXAMPLE_INFERRED_FILE[${example_name}]}" "../data/${example_name}/${variant}/${fold}/eval/${split}/${EXAMPLE_TARGET_FILE[${example_name}]}"
           # Set the target file.
           sed -i "s@${split_targets_path}@${split_targets_dir}/hotstart_target.txt@g" "${example_name}-eval.data"
         else
           # Ensure a previously failed run didn't dirty the target entry of eval.data file.
           sed -i "s@${split_targets_dir}/hotstart_target.txt@${split_targets_path}@g" "${example_name}-eval.data"
         fi

         ./run_offline.sh ${EXAMPLE_OPTIONS[${example_name}]} ${VARIANT_OPTIONS[${variant}_OFFLINE]} ${INITIALIZATION_OPTIONS[${initialization}]} > "${out_path}" 2> "${err_path}"

         # Save experiment output and parameters.
         mv "./inferred-predicates" "${out_directory}/inferred-predicates/${split}"
         cp "./${example_name}-eval.data" "${out_directory}/${split}/${example_name}-eval.data"
         cp "./${example_name}.psl" "${out_directory}/${split}/${example_name}.psl"
         cp "./run_offline.sh" "${out_directory}/${split}/run_offline.sh"

         if [ $initialization == "ATOM" ] && [ $split != "00" ]; then
           mv "${split_targets_dir}/hotstart_target.txt" "${out_directory}/${split}/hotstart_target.txt"
           # Reset the target file.
           sed -i "s@${split_targets_dir}/hotstart_target.txt@${split_targets_path}@g" "${example_name}-eval.data"
         fi

       popd > /dev/null
      fi
      prev_split=$split
    done
  done
}

function run_regret() {
  local example_name=$1
  local variant=$2
  local initialization=$3
  local fold=$4

  if [ $initialization == "RANDOM" ]; then
    return 0
  fi

  local out_path=""
  local err_path=""
  local out_directory=""
  local split_targets_dir=""
  local split_targets_path=""
  local experiment_options=""

  local inference_method="SGD_TI"
  local regretful_run_out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/online/NON_POWERSET/${initialization}"

  # Non_Powerset Regret.
  out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/regret/${inference_method}/${initialization}"

  # Set the inference method and logging leveel in the run_offline script.
  pushd . > /dev/null
   cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

   sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='.*'$/readonly ADDITIONAL_EVAL_OPTIONS='${INFERENCE_METHOD_OPTIONS[${inference_method}]} -D log4j.threshold=TRACE'/g" run_offline.sh
  popd > /dev/null

  for split in $(ls -d ./online-psl-examples/${example_name}/data/${example_name}/${variant}/${fold}/eval/*/ | cut -f 9 -d '/'); do
    echo "Running Regret Calculation PSL ${example_name}-${variant} (fold:${fold} -- time_step:${split})."

    # Declare paths to output files.
    out_path="${out_directory}/${split}/out.txt"
    err_path="${out_directory}/${split}/out.err"
    # Declare paths to target files.
    split_targets_dir="../data/${example_name}/${variant}/${fold}/eval/${split}"
    split_targets_path="${split_targets_dir}/${EXAMPLE_TARGET_FILE[${example_name}]}"

    if [[ -e "${out_path}" ]]; then
      echo "Output file already exists, skipping: ${out_path}"
    else
     mkdir -p "${out_directory}/${split}"
     [[ -d "${out_directory}/inferred-predicates" ]] || mkdir -p "${out_directory}/inferred-predicates"
     pushd . > /dev/null
       cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

       # Set the variant.
       sed -i "s@data/${example_name}/${example_name}.*/[0-9]\+/eval/@data/${example_name}/${variant}/00/eval/@g" "${example_name}-eval.data"
       # Set the data fold.
       sed -i "s/data\/${example_name}\/${variant}\/[0-9]\+/data\/${example_name}\/${variant}\/${fold}/g" "${example_name}-eval.data"
       # Set the data splits.
       sed -i "s/eval\/[0-9]\+/eval\/${split}/g" "${example_name}-eval.data"

       # cp inferred predicates from regretful run and split predicates to one file. Keep inferred predicates.
       cp "${regretful_run_out_directory}/inferred-predicates/${split}/${EXAMPLE_INFERRED_FILE[${example_name}]}" "${split_targets_dir}/regretful_target.txt"
       # Set the target file.
       sed -i "s@${split_targets_path}@${split_targets_dir}/regretful_target.txt@g" "${example_name}-eval.data"

       ./run_offline.sh -D sgd.maxiterations=3 -D inference.initialvalue=ATOM > "${out_path}" 2> "${err_path}"

       # Save experiment output and parameters.
       mv "./inferred-predicates" "${out_directory}/inferred-predicates/${split}"
       cp "./${example_name}-eval.data" "${out_directory}/${split}/${example_name}-eval.data"
       cp "./${example_name}.psl" "${out_directory}/${split}/${example_name}.psl"
       cp "./run_offline.sh" "${out_directory}/${split}/run_offline.sh"
       mv "${split_targets_dir}/regretful_target.txt" "${out_directory}/${split}/regretful_target.txt"
       # Reset the target file.
       sed -i "s@${split_targets_dir}/regretful_target.txt@${split_targets_path}@g" "${example_name}-eval.data"

     popd > /dev/null
    fi
  done

  # Non_Powerset Delta Model Gradient Calculation.
  out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/regret_delta_model/${inference_method}/${initialization}"

  # Set the inference method and the compute approximate delta gradient option in the run_offline script.
  experiment_options="${EXAMPLE_OPTIONS[${example_name}]} ${VARIANT_OPTIONS[${variant}_NON_POWERSET]} ${INITIALIZATION_OPTIONS[${initialization}]} -D inference.onlinecomputeapproximationdelta=true"

  echo "Running Approximation Delta Model Calculation PSL ${example_name}-${variant} (fold:${fold})."

  # Declare paths to output files.
  out_path="${out_directory}/out.txt"
  err_path="${out_directory}/out.err"

  if [[ -e "${out_path}" ]]; then
    echo "Output file already exists, skipping: ${out_path}"
  else
    mkdir -p ${out_directory}
    pushd . > /dev/null
       cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

       # Set the variant.
       sed -i "s@data/${example_name}/${example_name}.*/[0-9]\+/eval/@data/${example_name}/${variant}/00/eval/@g" "${example_name}-eval.data"
       # Set the data fold.
       sed -i "s/data\/${example_name}\/${variant}\/[0-9]\+/data\/${example_name}\/${variant}\/${fold}/g" "${example_name}-eval.data"
       # Set the data split.
       sed -i "s/eval\/[0-9]\+/eval\/00/g" "${example_name}-eval.data"
       # Set the commands location in run_client.sh.
       sed -i "s@^readonly COMMAND_FILE=.*'\$@readonly COMMAND_FILE='../data/${example_name}/${variant}/${fold}/eval/commands.txt'@g" run_client.sh
       # Ensure a previously failed offline run didn't dirty the target entry of eval.data file.
       sed -i "s@hotstart_target.txt@${EXAMPLE_TARGET_FILE[${example_name}]}@g" "${example_name}-eval.data"
       # Set the logging level.
       sed -i "s/^readonly ADDITIONAL_SERVER_OPTIONS='.*'$/readonly ADDITIONAL_SERVER_OPTIONS='-D log4j.threshold=TRACE'/g" run_server.sh
       sed -i "s/^readonly ADDITIONAL_CLIENT_OPTIONS='.*'$/readonly ADDITIONAL_CLIENT_OPTIONS='-D log4j.threshold=TRACE'/g" run_client.sh

       ./run.sh ${experiment_options} > "${out_path}" 2> "${err_path}"

       mv inferred-predicates "${out_directory}/"
       mv serverResponses "${out_directory}/"
       cp "./${example_name}-eval.data" "${out_directory}/${example_name}-eval.data"
       cp "./${example_name}.psl" "${out_directory}/${example_name}.psl"
       cp "./run_client.sh" "${out_directory}/run_client.sh"
       cp "./run_server.sh" "${out_directory}/run_server.sh"
       cp "./out_server.txt" "${out_directory}/out_server.txt"
       cp "./out_server.err" "${out_directory}/out_server.err"
       cp "./out_client.txt" "${out_directory}/out_client.txt"
       cp "./out_client.err" "${out_directory}/out_client.err"
     popd > /dev/null
  fi
}

function join_experiment_results() {
  local inferred_predicates_path=$1
  local split_targets_path=$2

  python3 "${BASE_DIR}"/join_experiment_results.py $(realpath ${inferred_predicates_path}) $(realpath ${split_targets_path})
}

function experiment_one() {
  local example_name=$1

  local out_directory=""
  local fold=""

  # Run over all folds
  for variant in ${EXAMPLE_VARIANT[${example_name}]}; do
    local variant_dir="${EXAMPLE_DIR}/${example_name}/data/${example_name}/${variant}"
    for fold_dir in $(ls -d ${variant_dir}/*); do
      for initialization in ${ATOM_INITIALIZATIONS}; do
        fold=$(basename ${fold_dir})
        # Timing Experiment
        run_online ${example_name} ${variant} ${initialization} ${fold} true
        run_offline ${example_name} ${variant} ${initialization} ${fold} true

        # Non-Timing Experiment
        run_online ${example_name} ${variant} ${initialization} ${fold} false
        run_offline ${example_name} ${variant} ${initialization} ${fold} false

        # Regret Experiment
        run_regret ${example_name} ${variant} ${initialization} ${fold}
      done
    done
  done
}

function main() {
  trap cleanup SIGINT

  for example_name in ${SUPPORTED_EXAMPLES} ; do
      # Run experiments
      experiment_one "${example_name}"
  done
}

function cleanup() {
  for pid in $(jobs -p); do
    pkill -P ${pid}
    kill ${pid}
  done
  pkill -P $$
  exit
}

main "$@"
