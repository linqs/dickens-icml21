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
EXAMPLE_OPTIONS[epinions]='-D sgd.maxiterations=500 -D reasoner.tolerance=1e-5f -D sgd.learningrate=0.1'

declare -A EXAMPLE_TARGET_FILE
EXAMPLE_TARGET_FILE[epinions]='trusts_target.txt'

declare -A EXAMPLE_INFERRED_FILE
EXAMPLE_INFERRED_FILE[epinions]='TRUSTS.txt'

declare -A EXAMPLE_VARIANT
EXAMPLE_VARIANT[epinions]='selected random'

declare -A INITIALIZATION_OPTIONS
INITIALIZATION_OPTIONS['ATOM']='-D inference.initialvalue=ATOM -D inference.onlinehotstart=true'
INITIALIZATION_OPTIONS['RANDOM']='-D inference.initialvalue=RANDOM -D inference.onlinehotstart=false'

readonly SUPPORTED_EXAMPLES='epinions'
readonly ATOM_INITIALIZATIONS='ATOM RANDOM'
readonly OFFLINE_INFERENCE_METHODS='SGD_TI SGD ADMM'
readonly ONLINE_GROUNDING_METHODS='NON_POWERSET'

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
    out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/online/${grounding_method}/${intialization}"
    out_path="${out_directory}/out.txt"
    err_path="${out_directory}/out.err"
    experiment_options="${EXAMPLE_OPTIONS[${example_name}]} ${INITIALIZATION_OPTIONS[${initialization}]}"

    if [[ -e "${out_path}" ]]; then
      echo "Output file already exists, skipping: ${out_path}"
    else
      mkdir -p ${out_directory}
      pushd . > /dev/null
         cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

         # Set the data fold.
         sed -i "s/data\/${example_name}\/[0-9]\+/data\/${example_name}\/${fold}/g" "${example_name}-eval.data"
         # Set the commands location in run_client.sh.
         sed -i "s@^readonly COMMAND_FILE=.*'\$@readonly COMMAND_FILE='../data/${example_name}/commands/${variant}-commands.txt'@g" run_client.sh
         # Ensure a previously failed offline run didn't dirty the target entry of eval.data file.
         sed -i "s@hotstart_target.txt@${EXAMPLE_TARGET_FILE[${example_name}]}@g" "${example_name}-eval.data"
         # Ensure a previously failed offline run didn't leave the wrong model file in the cli.
         # Set the logging level depending on whether this is timing experiment
         if [[ $timing == "false" ]]; then
          sed -i "s/^readonly ADDITIONAL_SERVER_OPTIONS='.*'$/readonly ADDITIONAL_SERVER_OPTIONS='-D log4j.threshold=TRACE'/g" run_server.sh
          sed -i "s/^readonly ADDITIONAL_CLIENT_OPTIONS='.*'$/readonly ADDITIONAL_CLIENT_OPTIONS='-D log4j.threshold=TRACE'/g" run_client.sh
         else
          sed -i "s/^readonly ADDITIONAL_SERVER_OPTIONS='.*'$/readonly ADDITIONAL_SERVER_OPTIONS=''/g" run_server.sh
          sed -i "s/^readonly ADDITIONAL_CLIENT_OPTIONS='.*'$/readonly ADDITIONAL_CLIENT_OPTIONS=''/g" run_client.sh
         fi
         cp "./selected_models/epinions-full.psl" "./epinions.psl"

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

  local out_path=""
  local err_path=""
  local out_directory=""
  local targets_dir="../data/${example_name}/${fold}/eval"
  local targets_path="${targets_dir}/${EXAMPLE_TARGET_FILE[${example_name}]}"
  local prev_model=""

  for inference_method in ${OFFLINE_INFERENCE_METHODS}; do
    out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/offline/${inference_method}/${initialization}"

    # Set the inference method in the run_offline script.
    pushd . > /dev/null
     cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

     sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='--infer.*'$/readonly ADDITIONAL_EVAL_OPTIONS='${INFERENCE_METHOD_OPTIONS[${inference_method}]}'/g" run_offline.sh
    popd > /dev/null

    # Run full model epxeriment.
    local model_name="${example_name}-full"
    echo "Running PSL ${example_name}-${variant} (fold:${fold} -- initialization:${initialization} -- model:${model_name})."
    # Declare paths to output files.
    out_path="${out_directory}/${model_name}/out.txt"
    err_path="${out_directory}/${model_name}/out.err"

    if [[ -e "${out_path}" ]]; then
      echo "Output file already exists, skipping: ${out_path}"
    else
     mkdir -p "${out_directory}/${model_name}"
     [[ -d "${out_directory}/inferred-predicates" ]] || mkdir -p "${out_directory}/inferred-predicates"
     pushd . > /dev/null
       cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

       # Set the data fold.
       sed -i "s/data\/${example_name}\/[0-9]\+/data\/${example_name}\/${fold}/g" "${example_name}-eval.data"

       # Ensure a previously failed run didn't dirty the target entry of eval.data file.
       sed -i "s@${targets_dir}/hotstart_target.txt@${targets_path}@g" "${example_name}-eval.data"

       # Copy the full model into cli for run offline
       cp "./selected_models/${model_name}.psl" "./${example_name}.psl"

       # Run offline for full epinions psl model
       ./run_offline.sh ${EXAMPLE_OPTIONS[${example_name}]} ${INITIALIZATION_OPTIONS[${initialization}]} > "${out_path}" 2> "${err_path}"

       # Save experiment output and parameters.
       mv "./inferred-predicates" "${out_directory}/inferred-predicates/${model_name}"
       cp "./${example_name}-eval.data" "${out_directory}/${model_name}/${example_name}-eval.data"
       cp "./${example_name}.psl" "${out_directory}/${model_name}/${example_name}.psl"
       cp "./run_offline.sh" "${out_directory}/${model_name}/run_offline.sh"
     popd > /dev/null
    fi

    prev_model="${example_name}-full"

    # Run perturbed model experiments.
    for model_file in $(ls "../online-psl-examples/${example_name}/cli/${variant}_models"); do
      echo "Running PSL ${example_name}-${variant} (fold:${fold} -- initialization:${initialization} -- model:${model_file})."

      local model_name="${model_file%.*}"

      # Declare paths to output files.
      out_path="${out_directory}/${model_name}/out.txt"
      err_path="${out_directory}/${model_name}/out.err"

      if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
      else
       mkdir -p "${out_directory}/${model_name}"
       [[ -d "${out_directory}/inferred-predicates" ]] || mkdir -p "${out_directory}/inferred-predicates"
       pushd . > /dev/null
         cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

         # Set the data fold.
         sed -i "s/data\/${example_name}\/[0-9]\+/data\/${example_name}\/${fold}/g" "${example_name}-eval.data"

         if [ $initialization == "ATOM" ]; then
           # Join inferred predicates and split predicates to one file. Keep inferred predicates.
           join_experiment_results "${out_directory}/inferred-predicates/${prev_model}/${EXAMPLE_INFERRED_FILE[${example_name}]}" "../data/${example_name}/${fold}/eval/${EXAMPLE_TARGET_FILE[${example_name}]}"
           # Set the target file.
           sed -i "s@${targets_path}@${targets_dir}/hotstart_target.txt@g" "${example_name}-eval.data"
         else
           # Ensure a previously failed run didn't dirty the target entry of eval.data file.
           sed -i "s@${targets_dir}/hotstart_target.txt@${targets_path}@g" "${example_name}-eval.data"
         fi

         # Copy the perturbed model into cli for run offline
         cp "./${variant}_models/${model_name}.psl" "./${example_name}.psl"

         ./run_offline.sh ${EXAMPLE_OPTIONS[${example_name}]} ${INITIALIZATION_OPTIONS[${initialization}]} > "${out_path}" 2> "${err_path}"

         # Save experiment output and parameters.
         mv "./inferred-predicates" "${out_directory}/inferred-predicates/${model_name}"
         cp "./${example_name}-eval.data" "${out_directory}/${model_name}/${example_name}-eval.data"
         cp "./${example_name}.psl" "${out_directory}/${model_name}/${example_name}.psl"
         cp "./run_offline.sh" "${out_directory}/${model_name}/run_offline.sh"

         if [ $initialization == "ATOM" ]; then
           mv "${targets_dir}/hotstart_target.txt" "${out_directory}/${model_name}/hotstart_target.txt"
           # Reset the target file.
           sed -i "s@${targets_dir}/hotstart_target.txt@${targets_path}@g" "${example_name}-eval.data"
         fi

       popd > /dev/null
      fi
      prev_model=${model_name}
    done
  done
}

function run_regret() {
  local example_name=$1
  local variant=$2
  local initialization=$3
  local fold=$4

  local out_path=""
  local err_path=""
  local out_directory=""
  local split_targets_dir=""
  local split_targets_path=""

  if [ $initialization == "RANDOM" ]; then
    return 0
  fi

  for inference_method in ${OFFLINE_INFERENCE_METHODS}; do
    out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/regret/${inference_method}/${initialization}"
    regretful_run_out_directory="${RESULTS_DIR}/${example_name}/${variant}/${fold}/online/NON_POWERSET/${initialization}"

    # Set the inference method in the run_offline script.
    pushd . > /dev/null
     cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

     sed -i "s/^readonly ADDITIONAL_EVAL_OPTIONS='--infer.*'$/readonly ADDITIONAL_EVAL_OPTIONS='${INFERENCE_METHOD_OPTIONS[${inference_method}]}'/g" run_offline.sh
    popd > /dev/null

    for split in $(ls "../online-psl-examples/${example_name}/cli/${variant}_models"); do
      echo "Running Regret Calculation PSL ${example_name}-${variant} (fold:${fold} -- initialization:${initialization} -- time_step:${split})."

      local model_name="${split%.*}"

      # Declare paths to output files.
      out_path="${out_directory}/${model_name}/out.txt"
      err_path="${out_directory}/${model_name}/out.err"
      # Declare paths to target files.
      split_targets_dir="../data/${example_name}/${fold}/eval"
      split_targets_path="${split_targets_dir}/${EXAMPLE_TARGET_FILE[${example_name}]}"

      if [[ -e "${out_path}" ]]; then
        echo "Output file already exists, skipping: ${out_path}"
      else
       mkdir -p "${out_directory}/${model_name}"
       [[ -d "${out_directory}/inferred-predicates" ]] || mkdir -p "${out_directory}/inferred-predicates"
       pushd . > /dev/null
         cd "${BASE_DIR}/../online-psl-examples/${example_name}/cli"

         # Set the data fold.
         sed -i "s/data\/${example_name}\/[0-9]\+/data\/${example_name}\/${fold}/g" "${example_name}-eval.data"

         # cp inferred predicates from regretful run and split predicates to one file. Keep inferred predicates.
         cp "${regretful_run_out_directory}/inferred-predicates/${model_name}/${EXAMPLE_INFERRED_FILE[${example_name}]}" "${split_targets_dir}/regretful_target.txt"
         # Set the target file.
         sed -i "s@${split_targets_path}@${split_targets_dir}/regretful_target.txt@g" "${example_name}-eval.data"

         ./run_offline.sh -D sgd.maxiterations=3 -D inference.initialvalue=ATOM > "${out_path}" 2> "${err_path}"

         # Save experiment output and parameters.
         mv "./inferred-predicates" "${out_directory}/inferred-predicates/${model_name}"
         cp "./${example_name}-eval.data" "${out_directory}/${model_name}/${example_name}-eval.data"
         cp "./${example_name}.psl" "${out_directory}/${model_name}/${example_name}.psl"
         cp "./run_offline.sh" "${out_directory}/${model_name}/run_offline.sh"
         mv "${split_targets_dir}/regretful_target.txt" "${out_directory}/${model_name}/regretful_target.txt"
         # Reset the target file.
         sed -i "s@${split_targets_dir}/regretful_target.txt@${split_targets_path}@g" "${example_name}-eval.data"

       popd > /dev/null
      fi
    done
  done
}

function join_experiment_results() {
  local inferred_predicates_path=$1
  local targets_path=$2

  python3 "${BASE_DIR}"/join_experiment_results.py $(realpath ${inferred_predicates_path}) $(realpath ${targets_path})
}

function experiment_one() {
  local example_name=$1

  local out_directory=""
  local fold=""

  # Run over all folds
  for variant in ${EXAMPLE_VARIANT[${example_name}]}; do
    local data_dir="${EXAMPLE_DIR}/${example_name}/data/${example_name}"
    for fold_dir in $(ls -d ${data_dir}/*); do
      for initialization in ${ATOM_INITIALIZATIONS}; do
        fold=$(basename ${fold_dir})
        if [[ "$fold" != "commands" ]]; then
          # Non-Timing Experiment
          run_online ${example_name} ${variant} ${initialization} ${fold} false
          run_offline ${example_name} ${variant} ${initialization} ${fold} false

          # Timing Experiment
          run_online ${example_name} ${variant} ${initialization} ${fold} true
          run_offline ${example_name} ${variant} ${initialization} ${fold} true

          run_regret ${example_name} ${variant} ${initialization} ${fold}
        fi
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
