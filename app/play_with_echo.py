import sys
import os
import os.path
from os import listdir
import random
from numpy import sqrt, asarray
from numpy.linalg import norm
from util import SKL_distance, bpm_distance, hasJsonDescriptor, key_distance, build_matrix_from_vector
import math
import time
import threading
import json
from operator import itemgetter
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject


# Configuration

# Use Phonos catalogue
phonos = False

# Set it to false for creative purpose ;)
avoid_repetitions = True

# Avoid repetitions must be set to True for this flag to have effects
force_different_consecutive_songs = True

# If avoid_repetitions, this sets the minimum hop in playlist between the same segment or the same song 
forbidden_value = 10

# Ratio of the maximum number of nearest neighbors in which we will look for similar segments
filter_size = 0.1

# Crossfade length (in seconds)
CROSSFADE = 0.8

# Determines when current player will stop playing current song
deadline = 0

# Length (in seconds) of automatic pause. Playback will automatically last for this amount of time, before being stopped for the same amount of time. Set to 0 for no pause.
automatic_pause_length = 0 * 60 
last_start = 0

# Initialization of global vars
currently_playing_song = None
previous_playing_song = None
current_track = None
current_artist = None
current_year = None
playlist = {}
songs_played = 0
global_volume = 1

# a traffic signal for managing the audio resources
green_light = True

# Gstreamer initialization
GObject.threads_init()
Gst.init(None)




class Player:
	"""
	A custom player based on gstreamer's playbin.
	"""

	def __init__(self, name):
		"""
		Creates a custom gstreamer playbin, sets initial values for its properties.
		"""

		self.name = name
		self.playmode = False
		self.playbin = Gst.ElementFactory.make("playbin")
		self.bus = self.playbin.get_bus()
		self.duration = 0
		self.seek_start = 0
		self.name_in_bold = ""
		self.forcedPause = False
		self.fade_in = False
		self.fade_out = False
		if self.name == "player1":
			self.name_in_bold = '\033[01;34m' + 'player1' + '\033[0m'

		else:
			self.name_in_bold = '\033[01;32m' + 'player2' + '\033[0m'


	def run(self, initial_sleep):
		"""
		Starts the player after a sleep of proper length.
		"""

		if initial_sleep != 0:
			computed_sleep = self.timeToSleepOnRun(initial_sleep)
		else:
			computed_sleep = 0

		# sleep and then play the assigned source file
		time.sleep(computed_sleep)

		self.play()
	

	def timeToSleepOnRun(self, input_time):
		"""
		The first time player2 is executed, it should sleep just until player1 starts fading out.
		"""

		return input_time - CROSSFADE
		

	def play(self):
		"""
		Forces the player to wait until its turns, grab next song and play it with fade in and fade out.
		"""

		while True:

			global green_light

			if (green_light and not self.forcedPause):

				green_light = False
				self.pick_next_song()
				self.set_playback()

				# PLAY
				global deadline
				deadline = time.time() + self.duration
				# fade in
				self.fadeIn(CROSSFADE)
				# actual playing
				time.sleep(self.duration - CROSSFADE)
				# fade out
				green_light = True # just before starting to fade out, let the other player know that this playback is over
				self.fadeOut(CROSSFADE)

				# if meanwhile the playback has been stopped, wait for setting new state up until it resumes
				if not self.forcedPause:
					self.set_pause()
					self.clean_queues()
				else:
					while True:
						if self.forcedPause:
							time.sleep(0.1)
						else:
							break
					self.set_pause()
					self.clean_queues()

			else:
				# wait for your turn
				time.sleep(.1)
				continue


	def pick_next_song(self):
		"""
		Pick the next song (if present) from the queue. Updates player's properties.
		"""

		while True:
			try:
				uri_from_queue = playlist[songs_played][0]
				title_from_queue = playlist[songs_played][1]
				artist_from_queue = playlist[songs_played][2]
				year_from_queue = playlist[songs_played][3]
				segment_interval = [playlist[songs_played][4], playlist[songs_played][5]]
				break
			except:
				time.sleep(.1)
				continue
		self.playbin.set_property("uri", uri_from_queue)
		self.uri = uri_from_queue
		self.title = title_from_queue
		self.artist = artist_from_queue
		self.year = year_from_queue
		self.seek_start = segment_interval[0]
		self.playbin.set_state(Gst.State.PAUSED)
		self.playbin.get_state(Gst.CLOCK_TIME_NONE)
		self.playbin.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, self.seek_start * Gst.SECOND)

		self.duration = segment_interval[1] - segment_interval[0]


	def set_playback(self):
		"""
		Sets everything ready for the playback.
		"""

		self.playbin.set_property("volume", 0.)
		self.playbin.set_state(Gst.State.PLAYING)
		self.playmode = True
		print self.name_in_bold, "playing", self.uri
		global currently_playing_song
		currently_playing_song = self.uri
		global current_artist
		current_artist = self.artist
		global current_track
		current_track = self.title
		global current_year
		current_year = self.year
		global songs_played
		songs_played += 1


	def set_pause(self):
		"""
		Handles the end of the playback.
		"""

		self.playbin.set_state(Gst.State.NULL)
		self.playmode = False


	def clean_queues(self):
		"""
		Delete useless items of playlist queue in order to avoid memory leaking.
		"""

		global songs_played
		for i in playlist.keys():
			if i < songs_played:
				del playlist[i]


	def pause(self):
		"""
		Immediately pauses the playback.
		"""

		self.forcedPause = True
		if self.playmode:
			self.playbin.set_state(Gst.State.PAUSED)


	def resume(self):
		"""
		Resumes the playback.
		"""

		self.forcedPause = False


	def fadeIn(self, duration):
		"""
		Performs a fade-in on current player.
		"""
		self.fade_in = True
		global global_volume
		actual_volume = 0
		while (actual_volume < round(global_volume, 1)):
			# print "Fade in:", actual_volume
			# print "global_volume:", global_volume
			time.sleep(duration/10.0)
			if global_volume != 0:
				actual_volume = round(actual_volume + global_volume/10., 2)
			else:
				actual_volume = 0
			self.playbin.set_property("volume", actual_volume)
		self.fade_in = False
	

	def fadeOut(self, duration):
		"""
		Performs a fade-out on current player.
		"""
		self.fade_out = True
		global global_volume
		actual_volume = global_volume
		while (actual_volume > 0):
			# print "Fade out:", actual_volume
			# print "global_volume:", global_volume
			time.sleep(duration/10.)
			if global_volume != 0:
				actual_volume = round(actual_volume - global_volume/10., 2)
			else:
				actual_volume = 0
			self.playbin.set_property("volume", actual_volume)		
		self.playbin.set_property("volume", 0)
		self.fade_out = False


	def setVolume(self, volume):
		if not self.fade_in and not self.fade_out:
			self.playbin.set_property("volume", volume)




class Play:
	"""
	Class that handles two different audio players in addition to the creation of the playlist according to the user interaction with the GUI.
	"""

	def __init__(self):

		self.player = Player("player1")
		self.player2 = Player("player2")

		self.slider1_value = 1
		self.slider2_value = 1
		self.slider3_value = 1
		self.slider4_value = 1
		self.slider5_value = 1
		self.slider6_value = 1
		self.slider7_value = 1
		self.slider8_value = 1
		self.slider9_value = 1
		self.slider10_value = 1

		self.loudness_upper_thresh = 0
		self.loudness_safe_upper_thresh = 0
		self.loudness_lower_thresh = 0
		self.loudness_safe_lower_thresh = 0
		self.dissonance_upper_thresh = 0
		self.dissonance_lower_thresh = 0
		self.dissonance_safe_upper_thresh = 0
		self.dissonance_safe_lower_thresh = 0
		self.onsetrate_upper_thresh = 0
		self.onsetrate_safe_upper_thresh = 0
		self.onsetrate_lower_thresh = 0
		self.onsetrate_safe_lower_thresh = 0
		self.barks_upper_thresh = 0
		self.barks_safe_upper_thresh = 0
		self.barks_lower_thresh = 0
		self.barks_safe_lower_thresh = 0
		self.acousticness_upper_thresh = 0
		self.acousticness_safe_upper_thresh = 0
		self.acousticness_lower_thresh = 0
		self.acousticness_safe_lower_thresh = 0
 
		self.ignoreSimilarities = False  # When set to true, we won't compute Kullback-Leibler distance to get a similarity score
		self.playing = False
		self.forbidden_songs = {}
		self.bars = 1
		self.forceAccept = False
		self.useLoudness = False
		self.useNoisiness = False
		self.useRhythm = False
		self.useBands = False
		self.useAcousticness = False
		self.last_start = 0
		self.on_automatic_pause = False
		self.slidersChanged = False


	def update_sliders(self, input1, input2, input3, input4, input5, input6, input7, input8, input9, input10):
		"""
		Function in account of managing values associated to sliders when the user interacts with them. 
		"""

		self.slider1_value = int(input1)
		self.slider2_value = int(input2)
		self.slider3_value = int(input3)
		self.slider4_value = int(input4)
		self.slider5_value = int(input5)
		self.slider6_value = int(input6)
		self.slider7_value = int(input7)
		self.slider8_value = int(input8)
		self.slider9_value = int(input9)
		self.slider10_value = int(input10)
		
		print self.slider1_value, self.slider2_value, self.slider3_value, self.slider4_value, self.slider5_value, \
			   self.slider6_value, self.slider7_value, self.slider8_value, self.slider9_value, self.slider10_value
		self.slidersChanged = True


	def activate_sliders(self, useLoudness, useNoisiness, useRhythm, useBands, useAcousticness):
		"""
		Function called when the user interacts with button for activating sliders. 
		"""

		if useLoudness == 'true':
			self.useLoudness = True
		else:
			self.useLoudness = False

		if useNoisiness == 'true':
			self.useNoisiness = True
		else:
			self.useNoisiness = False

		if useRhythm == 'true':
			self.useRhythm = True
		else:
			self.useRhythm = False

		if useBands == 'true':
			self.useBands = True
		else:
			self.useBands = False

		if useAcousticness == 'true':
			self.useAcousticness = True
		else:
			self.useAcousticness = False

		self.slidersChanged = True


	def changeSegmentlength(self, length):
		"""
		Function called when the user interacts with the segment length slider.
		"""

		self.bars = int(length)
		self.slidersChanged = True


	def update_rep_values(self, different_songs, different_segments):
		"""
		Function called when the user interacts with the advanced settings.
		"""
		global avoid_repetitions, force_different_consecutive_songs
		if different_segments == "false":
			avoid_repetitions = False
			force_different_consecutive_songs = False
		else:
			avoid_repetitions = True
			if different_songs == "true":
				force_different_consecutive_songs = True
		self.emptyUnplayedSongsQueue()
		self.ignoreSimilarities = True


	def play_request(self, playing):
		"""
		Handles user interaction with playback buttons. 
		"""
		if playing == "false":
			self.playing = False
			self.player.pause()
			self.player2.pause()
		else:
			global last_start
			last_start = time.time()
			self.playing = True
			self.player.resume()
			self.player2.resume()
			if self.on_automatic_pause:
				self.on_automatic_pause = False


	def update_volume(self, volume):
		global global_volume
		global_volume = int(volume) / 100.
		self.player.setVolume(global_volume)
		self.player2.setVolume(global_volume)


	def event_stream(self):
		"""
		Communicates a change in the playback to the user (a new song is playing).
		"""
		global current_track, current_artist, current_year

		previous_playing_song = None
		while True:
			if currently_playing_song != previous_playing_song:
				if current_track == "":
					current_track = "N/A"
				if current_artist == "":
					current_artist = "N/A"
				if current_year == "":
					current_year = "N/A"
				data_ = current_track + "/|" + current_artist + "/|" + current_year
				yield 'data: %s\n\n' % data_
				previous_playing_song = currently_playing_song
			time.sleep(.2)


	def communicate_automatic_pause(self):
		"""
		Communicates an automatic pause to the playback, so that the GUI can be updated (e.g. displaying the pause button instead of the playing one).
		"""
		previous_state = False
		while True:
			if self.on_automatic_pause != previous_state:
				yield 'data: %s\n\n' % self.on_automatic_pause
				previous_state = self.on_automatic_pause
			time.sleep(.2)


	def main(self):
		"""
		Loads the map for selected directory of music and calls the method for creating a playlist for it.
		"""

		if phonos:
			self.music_dir = "/home/giuseppe/empd/app/phonos/"
		else:
			self.music_dir = "/home/giuseppe/empd/app/music/"

		if len(sys.argv) > 1:
			self.music_dir = os.path.abspath(sys.argv[1])
		if self.music_dir[-1] != "/":
			self.music_dir += "/"

		try:
			self.beats_map, self._70s_songs, self._80s_songs, self._90s_songs, self._00s_songs, self._10s_songs = self.load_map()
			self.setupSlidersValues()
		except:
			print "There were some issues while loading the map of musical segments. Please build one for your collection."
			return
		self.createPlaylist()

		
	def createPlaylist(self):
		"""
		Randomly picks first song and creates a playlist for it.
		"""

		self.initializePlaylistCreation()
		firstSong = True
		global last_start
		last_start = time.time()

		# pick first song
		while True:
			last_segment_picked_index = random.randrange(len(self.beats_map))
			selected_segment = self.beats_map[last_segment_picked_index]
			start_time = float(selected_segment["start"])
			end_time = float(selected_segment["end"])
			selected_segment_length = end_time - start_time
			if selected_segment_length >= 2 * CROSSFADE:
				break
		last_song_picked = selected_segment["uri"]
		song_title = selected_segment["title"]
		song_artist = selected_segment["artist"]
		song_year = selected_segment["year"]
		first_segment_duration = selected_segment_length

		# put the selected song (or segment) into "forbidden_songs" queue, just to try to avoid to play it again very soon
		if force_different_consecutive_songs:
			self.updateForbiddenSongsQueue(last_song_picked)
		else:
			self.updateForbiddenSongsQueue(last_segment_picked_index)	

		# put the selected song into the playback queue
		playlist[self.counter] = ["file://" + last_song_picked, song_title, song_artist, song_year, start_time, end_time]

		# find all the other segments to be put in the playlist
		while True:
			self.perform_automatic_pause_if_necessary()
			self.checkForChanges()
			if self.deadlineIncoming():
				self.ignoreSimilarities = True

			# let the cpu rest (avoid overheating) if we still have to play more than 6 elements of the playlist
			if self.counter - songs_played >= 6:
				time.sleep(1)
				continue

			next_segment_index = self.pick_next_segment(last_segment_picked_index)
			next_segment = self.beats_map[next_segment_index]

			last_segment_picked_index = next_segment_index
			last_song_picked = next_segment["uri"]
			for i in range(self.bars):
				tmp_index = next_segment_index + i
				if last_song_picked == self.beats_map[tmp_index]["uri"]:
					next_segment_index_end = tmp_index
				else:
					break


			if force_different_consecutive_songs:
				self.updateForbiddenSongsQueue(last_song_picked)
			else:
				self.updateForbiddenSongsQueue(last_segment_picked_index)
			start_time = next_segment["start"]
			end_time = self.beats_map[next_segment_index_end]["end"]
			title = next_segment["title"]
			artist = next_segment["artist"]
			year = next_segment["year"]
			self.ignoreSimilarities = False
			self.counter += 1
			print "\033[01;36mSong added:", self.counter, last_song_picked, "interval:", [start_time, end_time], "duration:", end_time - start_time, "\033[0m"
			if firstSong:
				self.startPlayers(first_segment_duration)
				self.playing = True
				firstSong = False
			playlist[self.counter] = ["file://" + last_song_picked, title, artist, year, start_time, end_time]


	def pick_next_segment(self, current_segment_index):
		"""
		Finds a segment that observes sliders values and is quite similar to the last picked one.
		"""
		start_1 = time.time()
		# at first do some filtering based on sliders' values
		suitable_segments = self.find_segments_suitable_to_sliders()
		# print "len(suitable_segments) to sliders values:", len(suitable_segments)
		if len(suitable_segments) > 500:
			suitable_segments = self.random_subsampling(suitable_segments)

		# load features of last chosen segment
		current_segment = self.beats_map[current_segment_index]
		current_song_beats = self.load_json_descriptors(current_segment)
		current_bpm, current_loudness, current_mode = self.get_comparable_features(current_segment)
		bpm_thresh, loudness_thresh = self.get_thresholds()
		# filter the list by some simple audio properties (bpm, loudness, mode)
		filtered_suitable_segments = self.filter_list(suitable_segments, current_bpm, current_loudness, current_mode, bpm_thresh, loudness_thresh)
		# filtered candidates list could be empty
		if not filtered_suitable_segments:
			filtered_suitable_segments = suitable_segments

		print "Time to perform filtering:", time.time() - start_1
		current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar = self.load_mfccs(current_segment, current_song_beats)
		if len(filtered_suitable_segments) > 60:
			number_of_neighbors = self.select_number_of_neighbors(filtered_suitable_segments)
			neighbors = self.find_nearest_neighbors(filtered_suitable_segments, current_segment, number_of_neighbors)
			neighbors.sort()
			good_segments, analyzed_segments = self.get_suitable_segments_in_neighbors(neighbors, current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar, current_segment)
		else:
			good_segments, analyzed_segments = self.get_suitable_segments_in_neighbors(filtered_suitable_segments, current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar, current_segment)
		if good_segments:
			random_index = random.randrange(len(good_segments))
			return good_segments[random_index]
		else:
			analyzed_segments.sort(key=itemgetter('distance')) 
			random_index = random.randrange(math.ceil(len(analyzed_segments) / 10.))
			return analyzed_segments[random_index]["index"]


	def get_suitable_segments_in_neighbors(self, neighbors, current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar, current_segment):

		# keep track of all the segments analyzed (you may be forced to return of them if no segment is really similar to the current one)
		analyzed_segments = []
		# list of all segments whose timbre is really similar to the current one
		good_segments = []

		# this list will prevent us from loading the same .json file if we are analyzing a beat that is in the same song of the previous one
		target_song_beats = []
		previous_uri = ""
		for neighbor_index in neighbors:
			# avoid computing distance again if we have already computed that for this segment
			distance_previously_computed = self.neighbor_already_analyzed(neighbor_index, analyzed_segments)
			if distance_previously_computed: # equivalent to say "if this neighbor has already been analyzed"
				analyzed_segments.append({"index": neighbor_index, "distance": distance_previously_computed})
				continue

			neighbor = self.beats_map[neighbor_index]

			# if we don't meet the conditions to compute the skl distance, just compute an euclidean one between the points' coordinates in the map (it will require much less time)  
			if self.ignoreSimilarities:
				euclidean_distance = norm(asarray(current_segment["coords"]) - asarray(neighbor["coords"]))
				analyzed_segments.append({"index": neighbor_index, "distance": euclidean_distance})
				continue

			# now we need to load the descriptor file for the song to get the mfcc values
			if neighbor["uri"] != previous_uri: # load new json file if the beat we're analyzing is on a different song from the previous analyzed one
				target_song_beats = self.load_json_descriptors(neighbor)
				previous_uri = neighbor["uri"]

			neighbor_mfcc_mean, neighbor_mfcc_covar, neighbor_mfcc_invcovar = self.load_mfccs(neighbor, target_song_beats)
			skl_thresh = 20
			skl_dist = self.compute_skl_dist(current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar, neighbor_mfcc_mean, neighbor_mfcc_covar, neighbor_mfcc_invcovar)
			if skl_dist <= skl_thresh:
				good_segments.append(neighbor_index)
			else:
				analyzed_segments.append({"index": neighbor_index, "distance": skl_dist})

		return good_segments, analyzed_segments


	def random_subsampling(self, candidates_list):
		input_size = len(candidates_list)
		output_size = int(500 * (math.log(input_size, 500))**2.5)
		output_list = []
		for i in range(output_size):
			random_index = random.randrange(len(candidates_list))
			candidate = candidates_list[random_index]
			output_list.append(candidate)
		return output_list


	def load_mfccs(self, segment, song_beats):
		for beat in song_beats:
			if abs(segment["start"] - beat["others"][0]) <= 0.0001:
				segment_mfcc_mean = beat["mfcc.mean"]
				segment_mfcc_covar = beat["mfcc.covar"]
				segment_mfcc_invcovar = beat["mfcc.invcovar"]
				return segment_mfcc_mean, segment_mfcc_covar, segment_mfcc_invcovar


	def get_comparable_features(self, segment):
		segment_bpm = segment["bpm"]
		segment_loudness = segment["loudness"]
		segment_mode = segment["mode_fourth"]
		return segment_bpm, segment_loudness, segment_mode 


	def get_thresholds(self):
		bpm_thresh = 3 
		loudness_thresh = 5
		return bpm_thresh, loudness_thresh


	def filter_list(self, candidates_list, current_bpm, current_loudness, current_mode, bpm_thresh, loudness_thresh):
		output_list = candidates_list[:]
		for candidate_index in candidates_list:
			candidate_segment = self.beats_map[candidate_index]
			candidate_bpm, candidate_loudness, candidate_mode = self.get_comparable_features(candidate_segment)
			if not key_distance(current_mode, candidate_mode) or (bpm_distance(current_bpm, candidate_bpm) > bpm_thresh) or \
					(abs(current_loudness - candidate_loudness) > loudness_thresh):
				output_list.remove(candidate_index)
		return output_list


	def select_number_of_neighbors(self, candidates_list):
		if len(candidates_list) > len(self.beats_map):
			number_of_neighbors = int(filter_size * len(self.beats_map))
		else:
			number_of_neighbors = int(filter_size * len(candidates_list))
		return number_of_neighbors


	def compute_skl_dist(self, current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar, neighbor_mfcc_mean, neighbor_mfcc_covar, neighbor_mfcc_invcovar):
		dim = len(current_mfcc_mean)
		src_mean = asarray(current_mfcc_mean)
		src_covar = build_matrix_from_vector(current_mfcc_covar, dim)
		src_invcovar = build_matrix_from_vector(current_mfcc_invcovar, dim)
		tgt_mean = asarray(neighbor_mfcc_mean)
		tgt_covar = build_matrix_from_vector(neighbor_mfcc_covar, dim)
		tgt_invcovar = build_matrix_from_vector(neighbor_mfcc_invcovar, dim)
		skl_dist = SKL_distance(src_mean, tgt_mean, src_covar, tgt_covar, src_invcovar, tgt_invcovar, dim)
		return skl_dist


	def load_json_descriptors(self, segment):
		song_json_path = ('.').join(os.path.abspath(segment["uri"]).split('.')[:-1]) + ".json"
		with open(song_json_path) as data_file:
			song_json = json.load(data_file)
		return song_json["beats.descriptors"]


	def find_nearest_neighbors(self, candidates_list, src, number_of_neighbors):
		neighbors_indexes = []
		distances_to_point = []
		src_coords = src["coords"]	
		src_coords = asarray(src_coords)
		for tgt_index in candidates_list:
			tgt = self.beats_map[tgt_index]
			tgt_coords = tgt["coords"]
			tgt_coords = asarray(tgt_coords)
			# compute euclidean distance between point and tgt
			dist = norm(src_coords - tgt_coords) 
			distances_to_point.append({"id": tgt_index, "distance": dist})
		distances_to_point.sort(key=itemgetter('distance'))
		n_neighbors = []
		for i in range(number_of_neighbors):
			n_neighbors.append(distances_to_point[i]["id"]) 
		return n_neighbors


	def neighbor_already_analyzed(self, neighbor_index, analyzed_segments):
		for segment in analyzed_segments:
			if segment["index"] == neighbor_index:
				return segment["distance"]
		return None


	def find_segments_suitable_to_sliders(self):
		"""
		Returns a list of segments that observe sliders' values and whose length is greater than 2 * crossfade value.
		"""
		print "Loudness threshs:", self.loudness_lower_thresh, self.loudness_upper_thresh
		print "Noisiness threshs:", self.dissonance_lower_thresh, self.dissonance_upper_thresh
		print "Onsets threshs:", self.onsetrate_lower_thresh, self.onsetrate_upper_thresh
		print "Acousticness threshs:", self.acousticness_lower_thresh, self.acousticness_upper_thresh
		suitable_segments = []
		almost_suitable_segments = []
		random_segments = []
		pickFrom70s = self.slider1_value >= 1
		pickFrom80s = self.slider2_value >= 1
		pickFrom90s = self.slider3_value >= 1
		pickFrom00s = self.slider4_value >= 1
		pickFrom10s = self.slider5_value >= 1
		if pickFrom70s:
			for beat_index in self._70s_songs:
				beat = self.beats_map[beat_index]
				if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
					self.add_if_suitable(beat, self.slider1_value, beat_index, suitable_segments, almost_suitable_segments)
					random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
		if pickFrom80s:
			for beat_index in self._80s_songs:
				beat = self.beats_map[beat_index]
				if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
					self.add_if_suitable(beat, self.slider2_value, beat_index, suitable_segments, almost_suitable_segments)
					random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
		if pickFrom90s:
			for beat_index in self._90s_songs:
				beat = self.beats_map[beat_index]
				if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
					self.add_if_suitable(beat, self.slider3_value, beat_index, suitable_segments, almost_suitable_segments)
					random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
		if pickFrom00s:
			for beat_index in self._00s_songs:
				beat = self.beats_map[beat_index]
				if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
					self.add_if_suitable(beat, self.slider4_value, beat_index, suitable_segments, almost_suitable_segments)
					random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
		if pickFrom10s:
			for beat_index in self._10s_songs:
				beat = self.beats_map[beat_index]
				if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
					self.add_if_suitable(beat, self.slider5_value, beat_index, suitable_segments, almost_suitable_segments)
					random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
		if suitable_segments:
			print "Case #1"
			return suitable_segments
		elif almost_suitable_segments:
			print "Case #2"
			return almost_suitable_segments
		else:
			print "Case #3"
			random_segments.sort(key=itemgetter('distance'))
			segments_found = []
			for segment in random_segments[:int(math.ceil(len(random_segments)/1000.))]:
				segments_found.append(segment["index"])
			return segments_found


	def add_if_suitable(self, beat, slider_value, beat_index, suitable_segments, almost_suitable_segments):
		beat_loudness, beat_dissonance, beat_onsetrate, beat_barks, beat_acousticness, beat_uri = self.get_relevant_features(beat) 
		if self.slidersAreStrictlySatisfied(beat_loudness, beat_dissonance, beat_onsetrate, beat_barks, beat_acousticness):
			# print "beat", beat_index, "observes sliders:", beat_loudness, beat_dissonance, beat_onsetrate, beat_barks, beat_acousticness
			for i in range(slider_value):
				suitable_segments.append(beat_index)
		elif self.slidersAreAlmostSatisfied(beat_loudness, beat_dissonance, beat_onsetrate, beat_barks, beat_acousticness):
			for i in range(slider_value):
				almost_suitable_segments.append(beat_index)


	def not_a_repetition(self, beat, beat_index):
		return not avoid_repetitions or ((force_different_consecutive_songs and beat["uri"] not in self.forbidden_songs.keys()) or \
			(not force_different_consecutive_songs and beat_index not in self.forbidden_songs.keys()))


	def distance_to_slider_conf(self, beat):
		beat_loudness = beat["loudness"]
		beat_dissonance = beat["dissonance"]
		beat_onsetrate = beat["onsetrate"]
		beat_acousticness = beat["acousticness"]
		loudness_distance = beat_loudness - self.loudness_lower_thresh
		dissonance_distance = beat_dissonance - self.dissonance_lower_thresh
		onsetrate_distance = beat_onsetrate - self.onsetrate_lower_thresh
		acoustic_distance = beat_acousticness - self.acousticness_lower_thresh
		dist = (loudness_distance)**2 + (dissonance_distance)**2 + (onsetrate_distance)**2 + (acoustic_distance)**2
		return sqrt(dist)


	def deadlineIncoming(self):
		deadline_incoming = (len(playlist) <= 1) and (deadline - time.time() < 5)
		return deadline_incoming


	def load_map(self):
		with open(self.music_dir + "outputmap") as data_file:
			json_content = json.load(data_file)
		return json_content["all_beats"], json_content["70s"], json_content["80s"], json_content["90s"], json_content["00s"], json_content["10s"]


	def get_relevant_features(self, beat):
		beat_loudness = beat["loudness"]
		beat_dissonance = beat["dissonance"]
		beat_onsetrate = beat["onsetrate"]
		beat_barks = beat["barks"]
		beat_acousticness = beat["acousticness"]
		beat_uri = beat["uri"]
		return beat_loudness, beat_dissonance, beat_onsetrate, beat_barks, beat_acousticness, beat_uri


	def setupSlidersValues(self):
		with open(self.music_dir + "/sliders_values") as data_file:
			json_content = json.load(data_file)
		self.loudness_low = float(json_content["loudness"][0])
		self.loudness_1st = float(json_content["loudness"][1])
		self.loudness_2nd = float(json_content["loudness"][2])
		self.loudness_3rd = float(json_content["loudness"][3])
		self.loudness_high = float(json_content["loudness"][4])
		self.dissonance_low = float(json_content["dissonance"][0])
		self.dissonance_1st = float(json_content["dissonance"][1])
		self.dissonance_2nd = float(json_content["dissonance"][2])
		self.dissonance_3rd = float(json_content["dissonance"][3])
		self.dissonance_high = float(json_content["dissonance"][4])
		self.onsets_low = float(json_content["onsets"][0])
		self.onsets_1st = float(json_content["onsets"][1])
		self.onsets_2nd = float(json_content["onsets"][2])
		self.onsets_3rd = float(json_content["onsets"][3])
		self.onsets_high = float(json_content["onsets"][4])
		self.bark_0 = float(json_content["bark"][0])
		self.bark_1 = float(json_content["bark"][1])
		self.bark_2 = float(json_content["bark"][2])
		self.bark_3 = float(json_content["bark"][3])
		self.bark_4 = float(json_content["bark"][4])
		self.bark_5 = float(json_content["bark"][5])
		self.bark_6 = float(json_content["bark"][6])
		self.bark_7 = float(json_content["bark"][7])
		self.bark_8 = float(json_content["bark"][8])
		self.bark_9 = float(json_content["bark"][9])
		self.bark_10 = float(json_content["bark"][10])
		self.bark_11 = float(json_content["bark"][11])
		self.bark_12 = float(json_content["bark"][12])
		self.bark_13 = float(json_content["bark"][13])
		self.bark_14 = float(json_content["bark"][14])
		self.bark_15 = float(json_content["bark"][15])
		self.bark_16 = float(json_content["bark"][16])
		self.bark_17 = float(json_content["bark"][17])
		self.bark_18 = float(json_content["bark"][18])
		self.bark_19 = float(json_content["bark"][19])
		self.bark_20 = float(json_content["bark"][20])
		self.bark_21 = float(json_content["bark"][21])
		self.bark_22 = float(json_content["bark"][22])
		self.bark_23 = float(json_content["bark"][23])
		self.bark_24 = float(json_content["bark"][24])
		self.bark_25 = float(json_content["bark"][25])
		self.bark_26 = float(json_content["bark"][26])
		self.acousticness_low = float(json_content["acousticness"][0])
		self.acousticness_1st = float(json_content["acousticness"][1])
		self.acousticness_2nd = float(json_content["acousticness"][2])
		self.acousticness_3rd = float(json_content["acousticness"][3])
		self.acousticness_high = float(json_content["acousticness"][4])


	def adaptThresholdsToSliders(self):
		"""
		Update thresholds according to sliders' values.
		"""

		if self.slider6_value == 1:
			self.loudness_lower_thresh = self.loudness_low
			self.loudness_upper_thresh = self.loudness_1st
			self.loudness_safe_lower_thresh = self.loudness_low
			self.loudness_safe_upper_thresh = self.loudness_2nd
		elif self.slider6_value == 2:
			self.loudness_lower_thresh = self.loudness_1st
			self.loudness_upper_thresh = self.loudness_2nd
			self.loudness_safe_lower_thresh = self.loudness_low	
			self.loudness_safe_upper_thresh = self.loudness_2nd
		elif self.slider6_value == 3:
			self.loudness_lower_thresh = self.loudness_2nd
			self.loudness_upper_thresh = self.loudness_3rd
			self.loudness_safe_lower_thresh = self.loudness_1st
			self.loudness_safe_upper_thresh = self.loudness_3rd
		else:
			self.loudness_lower_thresh = self.loudness_3rd
			self.loudness_upper_thresh = self.loudness_high
			self.loudness_safe_lower_thresh = self.loudness_2nd
			self.loudness_safe_upper_thresh = self.loudness_high

		if self.slider7_value == 1:
			self.dissonance_lower_thresh = self.dissonance_low
			self.dissonance_upper_thresh = self.dissonance_1st
			self.dissonance_safe_lower_thresh = self.dissonance_low
			self.dissonance_safe_upper_thresh = self.dissonance_2nd
		elif self.slider7_value == 2:
			self.dissonance_lower_thresh = self.dissonance_1st
			self.dissonance_upper_thresh = self.dissonance_2nd
			self.dissonance_safe_lower_thresh = self.dissonance_low
			self.dissonance_safe_upper_thresh = self.dissonance_2nd
		elif self.slider7_value == 3:
			self.dissonance_lower_thresh = self.dissonance_2nd
			self.dissonance_upper_thresh = self.dissonance_3rd
			self.dissonance_safe_lower_thresh = self.dissonance_1st
			self.dissonance_safe_upper_thresh = self.dissonance_3rd
		else:
			self.dissonance_lower_thresh = self.dissonance_3rd
			self.dissonance_upper_thresh = self.dissonance_high
			self.dissonance_safe_lower_thresh = self.dissonance_2nd
			self.dissonance_safe_upper_thresh = self.dissonance_high

		if self.slider8_value == 1:
			self.onsetrate_lower_thresh = self.onsets_low
			self.onsetrate_upper_thresh = self.onsets_1st
			self.onsetrate_safe_lower_thresh = self.onsets_low
			self.onsetrate_safe_upper_thresh = self.onsets_2nd
		elif self.slider8_value == 2:
			self.onsetrate_lower_thresh = self.onsets_1st
			self.onsetrate_upper_thresh = self.onsets_2nd
			self.onsetrate_safe_lower_thresh = self.onsets_low
			self.onsetrate_safe_upper_thresh = self.onsets_2nd
		elif self.slider8_value == 3:
			self.onsetrate_lower_thresh = self.onsets_2nd
			self.onsetrate_upper_thresh = self.onsets_3rd
			self.onsetrate_safe_lower_thresh = self.onsets_1st
			self.onsetrate_safe_upper_thresh = self.onsets_3rd
		else:
			self.onsetrate_lower_thresh = self.onsets_3rd
			self.onsetrate_upper_thresh = self.onsets_high
			self.onsetrate_safe_lower_thresh = self.onsets_2nd
			self.onsetrate_safe_upper_thresh = self.onsets_high

		if self.slider9_value == 1:
			self.barks_lower_thresh = 0
			self.barks_upper_thresh = 4	
			self.barks_safe_lower_thresh = 0
			self.barks_safe_upper_thresh = 10
		elif self.slider9_value == 2:
			self.barks_lower_thresh = 4
			self.barks_upper_thresh = 8	
			self.barks_safe_lower_thresh = 0
			self.barks_safe_upper_thresh = 12
		elif self.slider9_value == 3:
			self.barks_lower_thresh = 8
			self.barks_upper_thresh = 14		
			self.barks_safe_lower_thresh = 4
			self.barks_safe_upper_thresh = 18	
		elif self.slider9_value == 4:
			self.barks_lower_thresh = 14
			self.barks_upper_thresh = 27
			self.barks_safe_lower_thresh = 10
			self.barks_safe_upper_thresh = 27

		if self.slider10_value == 1:
			self.acousticness_lower_thresh = self.acousticness_low
			self.acousticness_upper_thresh = self.acousticness_1st
			self.acousticness_safe_lower_thresh = self.acousticness_low
			self.acousticness_safe_upper_thresh = self.acousticness_2nd
		elif self.slider10_value == 2:
			self.acousticness_lower_thresh = self.acousticness_1st
			self.acousticness_upper_thresh = self.acousticness_2nd
			self.acousticness_safe_lower_thresh = self.acousticness_low
			self.acousticness_safe_upper_thresh = self.acousticness_2nd
		elif self.slider10_value == 3:
			self.acousticness_lower_thresh = self.acousticness_2nd
			self.acousticness_upper_thresh = self.acousticness_3rd
			self.acousticness_safe_lower_thresh = self.acousticness_1st
			self.acousticness_safe_upper_thresh = self.acousticness_3rd
		elif self.slider10_value == 4:
			self.acousticness_lower_thresh = self.acousticness_3rd
			self.acousticness_upper_thresh = self.acousticness_high
			self.acousticness_safe_lower_thresh = self.acousticness_2nd
			self.acousticness_safe_upper_thresh = self.acousticness_high


	def slidersAreStrictlySatisfied(self, actual_loudness, actual_dissonance, actual_onsetrate, bark_bands_over_thresh, song_acousticness):
		loudness_satisfied = not self.useLoudness or \
			(actual_loudness <= self.loudness_upper_thresh and actual_loudness >= self.loudness_lower_thresh)
		noisiness_satisfied = not self.useNoisiness or (actual_dissonance <= self.dissonance_upper_thresh and actual_dissonance >= self.dissonance_lower_thresh)
		onsetrate_satisfied = not self.useRhythm or (actual_onsetrate <= self.onsetrate_upper_thresh and actual_onsetrate >= self.onsetrate_lower_thresh)
		barks_satisfied = not self.useBands or (bark_bands_over_thresh <= self.barks_upper_thresh and bark_bands_over_thresh >= self.barks_lower_thresh)
		acousticness_satisfied = not self.useAcousticness or (song_acousticness <= self.acousticness_upper_thresh and song_acousticness >= self.acousticness_lower_thresh)		
		
		strictSimilarity = loudness_satisfied and noisiness_satisfied and onsetrate_satisfied and barks_satisfied and acousticness_satisfied
		return strictSimilarity


	def slidersAreAlmostSatisfied(self, actual_loudness, actual_dissonance, actual_onsetrate, bark_bands_over_thresh, song_acousticness):
		loudness_almost_satisfied = not self.useLoudness or \
			(actual_loudness <= self.loudness_safe_upper_thresh and actual_loudness >= self.loudness_safe_lower_thresh)

		noisiness_almost_satisfied = not self.useNoisiness or \
			(actual_dissonance <= self.dissonance_safe_upper_thresh and actual_dissonance >= self.dissonance_safe_lower_thresh)

		onsetrate_almost_satisfied = not self.useRhythm or \
			(actual_onsetrate <= self.onsetrate_safe_upper_thresh and actual_onsetrate >= self.onsetrate_safe_lower_thresh)

		barks_almost_satisfied = not self.useBands or \
			(bark_bands_over_thresh <= self.barks_safe_upper_thresh and bark_bands_over_thresh >= self.barks_safe_lower_thresh)

		acousticness_almost_satisfied = not self.useAcousticness or (song_acousticness <= self.acousticness_safe_upper_thresh and \
			song_acousticness >= self.acousticness_safe_lower_thresh)

		approxSimilarity = loudness_almost_satisfied and noisiness_almost_satisfied and onsetrate_almost_satisfied and \
							barks_almost_satisfied and acousticness_almost_satisfied
		return approxSimilarity


	def updateSliders(self):
		"""
		If all the decades sliders have the same value, just put them to 1 (so that weighted queue will be shorter)
		"""
		if (self.slider1_value == self.slider2_value) and (self.slider2_value == self.slider3_value) and (self.slider3_value == self.slider4_value) and \
			(self.slider4_value == self.slider5_value):
			self.slider1_value = 1
			self.slider2_value = 1
			self.slider3_value = 1
			self.slider4_value = 1
			self.slider5_value = 1


	def checkForChanges(self):
		"""
		Check if some sliders have been changed by the user. 
		If so, make proper changes to values.
		"""
		if self.slidersChanged:
			self.adaptThresholdsToSliders()
			self.updateSliders()
			self.emptyUnplayedSongsQueue()
			self.ignoreSimilarities = True
			self.slidersChanged = False


	def emptyUnplayedSongsQueue(self):
		"""
		Empties the playlist queue.
		"""
		# empty the queue of songs that haven't been played yet
		if len(playlist) >= 1:
			for i in playlist.keys():
				if i >= songs_played - 1:
					del playlist[i]
		self.counter = songs_played - 1


	def updateForbiddenSongsQueue(self, candidate_song):
		"""
		Updates values in forbidden songs queue (that is the list of songs that cannot be selected because they have been used recently).
		Reduces by 1 the number of turns on which they will still be forbidden.
		"""
		for key in self.forbidden_songs.keys():
			val = self.forbidden_songs[key]
			val -= 1
			if val == 0:
				self.forbidden_songs.pop(key, None)
			else:
				self.forbidden_songs.update({key:val})
		self.forbidden_songs.update({candidate_song: forbidden_value})


	def perform_automatic_pause_if_necessary(self):
		"""
		Automatically performs a stop of 'automatic_pause_length' seconds each 'automatic_pause_length' seconds.
		"""
		global last_start
		if self.playing and automatic_pause_length != 0 and (time.time() - last_start > automatic_pause_length):
			self.playing = False
			self.on_automatic_pause = True
			self.player.pause()
			self.player2.pause()
			need_to_restart = True
			for i in range(automatic_pause_length * 10):
				time.sleep(0.1)
				if self.playing: # check if the user have resumed the playback manually
					need_to_restart = False
					break
			if need_to_restart:
				global last_start
				last_start = time.time()
				self.playing = True
				self.on_automatic_pause = False
				self.player.resume()
				self.player2.resume()


	def initializePlaylistCreation(self):
		"""
		Initialize variables so that createPlaylist() can run properly.
		"""
		global songs_played, green_light
		songs_played = 0
		green_light = True
		self.counter = 0


	def startPlayers(self, player2_initialsleep):
		"""
		Starts the two audio players, that will be playing together in order to achieve a crossfade effect.
		"""
		thr = threading.Thread(target = self.player.run, args = (0,))
		thr2 = threading.Thread(target = self.player2.run, args = (player2_initialsleep,))
		thr.start()
		thr2.start()