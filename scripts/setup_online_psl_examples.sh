#!/usr/bin/env bash

# Note that you can change the version of PSL used with the PSL_VERSION option in the run inference and run wl scripts.
readonly BASE_DIR=$(realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..)
readonly ONLINE_PSL_EXAMPLES_DIR="${BASE_DIR}/online-psl-examples"

readonly PSL_VERSION='2.3.0-SNAPSHOT'

# Note that for macOS the following two commands will not work.
# Instead comment out the two commands and uncomment the `readonly JAVA_MEM_GB=24`.
readonly AVAILABLE_MEM_KB=$(cat /proc/meminfo | grep 'MemTotal' | sed 's/^[^0-9]\+\([0-9]\+\)[^0-9]\+$/\1/')
readonly JAVA_MEM_GB=$((${AVAILABLE_MEM_KB} / 1024 / 1024 / 5 * 5 - 5))
#readonly JAVA_MEM_GB=24

# Common to all examples.
function standard_setup() {
    for exampleDir in `find ${ONLINE_PSL_EXAMPLES_DIR} -maxdepth 1 -mindepth 1 -type d -not -name '.*' -not -name '_scripts'`; do
        pushd . > /dev/null
            cd "${exampleDir}/cli"

            # Increase memory allocation.
            sed -i "s/java -jar/java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar/" run_offline.sh
            sed -i "s/java -jar/java -Xmx${JAVA_MEM_GB}G -Xms${JAVA_MEM_GB}G -jar/" run_server.sh
            sed -i "s/java -jar/java -Xmx8G -Xms8G -jar/" run_client.sh

            # Set the PSL version.
            sed -i "s/^readonly PSL_VERSION='.*'$/readonly PSL_VERSION='${PSL_VERSION}'/" run_offline.sh
            sed -i "s/^readonly PSL_VERSION='.*'$/readonly PSL_VERSION='${PSL_VERSION}'/" run_server.sh
            sed -i "s/^readonly PSL_VERSION='.*'$/readonly PSL_VERSION='${PSL_VERSION}'/" run_client.sh

            # Copy the PSL resources jar into the CLI and deactivate fetch psl command.
            cp ../../../psl-resources/psl-cli-2.3.0-SNAPSHOT.jar ./
            sed -i 's/^\(\s\+\)fetch_psl/\1# fetch_psl/' run_offline.sh
            sed -i 's/^\(\s\+\)fetch_psl/\1# fetch_psl/' run_server.sh
            sed -i 's/^\(\s\+\)fetch_psl/\1# fetch_psl/' run_client.sh
        popd > /dev/null
    done
}

function individual_setup() {
    # Fetch Data Script
    for exampleDir in `find ${ONLINE_PSL_EXAMPLES_DIR} -maxdepth 1 -mindepth 1 -type d -not -name '.*' -not -name '_scripts'`; do
        pushd . > /dev/null
          cd "${exampleDir}/data"

          local fetchDataScript="./fetchData.sh"
          if [ ! -e "${fetchDataScript}" ]; then
              continue
          fi

          ${fetchDataScript}
        popd > /dev/null
    done
}

function main() {
   trap exit SIGINT

   standard_setup
   individual_setup

   exit 0
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
