from app import app
from flask import render_template, request, Response
from app import play_with_echo
import threading
import gevent
import time

play = play_with_echo.Play()

@app.route('/')
def index():
	thr = threading.Thread(target = play.main, args = ())
	thr.start()
	return render_template('index.html')

@app.route('/stream', methods=['GET'])
def stream(): 
    return Response(play.event_stream(), mimetype="text/event-stream")

@app.route('/autopause', methods=['GET'])
def autopause(): 
    return Response(play.communicate_automatic_pause(), mimetype="text/event-stream")

@app.route('/loading', methods=['GET'])
def loading(): 
    return Response(play.loading(), mimetype="text/event-stream")

@app.route('/updateslider', methods=['POST'])
def updateSliders():
	play.update_sliders(request.form['slider1'], request.form['slider2'], request.form['slider3'], request.form['slider4'],\
		request.form['slider5'], request.form['slider6'], request.form['slider7'], request.form['slider8'], request.form['slider9'], request.form['slider10'])
	return request.form['slider1']

@app.route('/back', methods=['POST'])
def onExit():
	func = request.environ.get('werkzeug.server.shutdown')
	if func is None:
	    raise RuntimeError('Not running with the Werkzeug Server')
	func()
	print "Shutting down the server"
	return "Shutting down..."

@app.route('/volume', methods=['POST'])
def updateVolume():
	play.update_volume(request.form['volume'])
	return request.form['volume']

@app.route('/repsegments', methods=['POST'])
def updateRepValues():
	play.update_rep_values(request.form['different_songs'], request.form['different_segments'])
	return request.form['different_songs']

@app.route('/usesliders', methods=['POST'])
def activateSliders():
	play.activate_sliders(request.form['loudness'], request.form['noisiness'], request.form['sparseness'], request.form['bass'], request.form['high'])

	return request.form['loudness']

@app.route('/segmentlength', methods=['POST'])
def changelength():
	play.changeSegmentlength(request.form['length'])

	return request.form['length']

@app.route('/play', methods=['POST'])
def currentlyPlaying():
	play.play_request(request.form['playing'])

	return request.form['playing']

@app.route('/stream')
def getSong():
	return Response(play_with_echo.songChanged(), mimetype="text/event-stream")
