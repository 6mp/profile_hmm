#!/bin/bash

for index in `seq 1 3`; do
    DIR="examples/test${index}"

    buildDir/profile_hmm \
        -m $DIR/model.hmm \
        -q $DIR/queries.fas \
        -o $DIR/output.txt

    diff $DIR/scores.txt $DIR/output.txt
done

