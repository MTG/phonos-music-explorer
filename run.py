#!flask/bin/python
from app import app
from gevent.wsgi import WSGIServer

# app.debug = True
# server = WSGIServer(("", 5000), app)
# server.serve_forever()
app.run('0.0.0.0', debug = True, port = 5000, threaded = True)
