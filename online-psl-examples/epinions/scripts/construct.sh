#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly DATA_CONSTRUCTION_SCRIPT="${THIS_DIR}/data-construction/construct.py"

function main() {
   trap exit SIGINT

   python3 ${DATA_CONSTRUCTION_SCRIPT} ${DATA_DIR}
}

main "$@"
