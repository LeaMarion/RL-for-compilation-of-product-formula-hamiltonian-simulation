#!/bin/bash

echo "Enter algorithm name (DDQN, SA or MCTS)"
read ALGO

echo "Range of values (ex. 0 4)"
read MIN MAX

echo "$ALGO for $MIN to $MAX"

if [ $ALGO == "DDQN" ]
then
  for i in $(seq $MIN $MAX); do python3 HSC_processing_script.py $i; done
elif [ $ALGO == "SA" ]
then
  for i in $(seq $MIN $MAX); do python3 HSC_SA_processing_script.py $i; done
elif [ $ALGO == "MCTS" ]
then
  for i in $(seq $MIN $MAX); do python3 HSC_MCTS_processing_script.py $i; done
else
  echo "Algorithm $ALGO not recognized"
fi