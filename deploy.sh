#!/usr/bin/env bash
if [ $# -lt 2 ]; then
    if [ $# -lt 1 ]; then
	    server='simrem'
	    # extracmd="ssh uni2 'cd ~/mclnet/; ./deploy.sh uni3; ./deploy.sh uni4'"
  	    # extracmd="../mclnet/deploy.sh $server"
  	    extracmd="echo $server done"
	else
	    server=$1
	    extracmd="echo $server done"
	fi
	rsync -avz --no-links --exclude-from rsync_exclude.txt ./ $server:~/code/GibsonSim2RealChallenge/
	rsync -avz --no-links --exclude-from rsync_exclude.txt ../GibsonEnvV2/ $server:~/code/GibsonSim2RealChallenge/GibsonEnvV2/
    rsync -avz --no-links --exclude-from ../mclnet/rsync_exclude.txt ../mclnet/ $server:~/mclnet/
    eval $extracmd
else
    echo "Not supported"
fi



