#!/usr/bin/env bash
if [ $# -eq 2 ]; then
    server=$1
    folder=$2
    targetdir="./temp/evals/"

    # --include="*.png" --exclude="*.chk*" --exclude="final*" --exclude="checkpoint*"
    rsync -avz --size-only  --exclude="*.chk*" "$server:~/code/GibsonSim2RealChallenge/$folder/*" $targetdir
    echo "done"

else
    if [ $# -lt 1 ]; then
        # extracmd="ssh uni1 'cd ~/mclnet/; ./deploy.sh uni2; ./deploy.sh uni3; ./deploy.sh uni4; ./deploy.sh unip; ./deploy.sh wgpu'"
        extracmd="./getevalresults.sh simrem"
    else
        server=$1
        extracmd="echo $server done"
    fi
    targetdir="./temp/evals/"
    rsync -avz --size-only  --exclude="*.tfrecords*" "$server:~/code/GibsonSim2RealChallenge/temp/evals/*" $targetdir

    eval $extracmd
fi
