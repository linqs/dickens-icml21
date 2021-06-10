#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG

function main() {
   trap cleanup SIGINT SIGTERM

   # Run PSL
   runAll "$@"
}

function runAll() {
   echo "Running Online Server"
   ./run_server.sh "$@" > out_server.txt 2> out_server.err &
   local server_pid=$!

   echo "Running Online Client"
   ./run_client.sh "$@" > out_client.txt 2> out_client.err

   echo "Waiting on Online Server"
   wait ${server_pid}
   echo "Finished Waiting on Online Server"
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
