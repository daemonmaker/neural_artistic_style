if [ $z "$1" ]
        then
		celery -A $1.Server:celery worker --loglevel=info >/logs/celery-log.txt 2>&1 &
else
        echo "Implementation name required. Options are [\"gatys\"]."
fi
