if [ $z "$1" ]
	then
		FLASK_APP=/tf/$1/Server.py flask run --host=0.0.0.0 >/logs/flask-log.txt 2>&1
else
	echo "Implementation name required. Options are [\"gatys\"]."	
fi
