#!flask/bin/python
from app import app
from multiprocessing import Process

# disable verbose mode of flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# app.run('0.0.0.0', debug = False, port = 5000, threaded = True)

def run_server():
	print "\033[01;31mServer started, listening at port 5000\033[0m"
	app.run('0.0.0.0', debug = True, port = 5000, threaded = True)

while True:
	server = Process(target=run_server)
	server.start()
	server.join()

