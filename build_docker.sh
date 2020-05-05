#!/usr/bin/env bash

rm -rf ./build
mkdir ./build
rsync -avz --no-links --exclude-from ~/mclnet/rsync_exclude.txt ~/mclnet ./build/


docker build . -t my_submission_clean2