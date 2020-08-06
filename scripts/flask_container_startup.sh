#! /bin/bash

if [ $z "$1" ]
	then
		/scripts/redis_startup.sh
		/scripts/celery_startup.sh $1
		/scripts/flask_startup.sh $1
else
	echo "Implementation name required. Options are [\"gatys\"]."
fi
