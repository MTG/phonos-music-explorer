import os.path
# import math
# from scipy.spatial.distance import mahalanobis
import json
# from scipy import asarray
from numpy.linalg import inv, det
from numpy import dot, trace, subtract, diag, sqrt, reshape, asarray
#import numpy.matrix.sum as matrix_sum
from math import log
import numpy

def hasDescriptorFile(filename):
	return os.path.isfile(filename + ".sig")


def hasJsonDescriptor(filename):
	found = os.path.isfile(('.').join(filename.split('.')[:-1]) + ".json")
	return found


def SKL_distance(mfcc_mean_0, mfcc_mean_1, mfcc_covar_0, mfcc_covar_1, mfcc_inv_covar_0, mfcc_inv_covar_1, dim = 13):
	"""
	Computes symmetric Kullback-Leibler divergence
	"""

	first_distance = KL_distance(mfcc_mean_0, mfcc_mean_1, mfcc_covar_0, mfcc_covar_1, mfcc_inv_covar_1, dim)
	second_distance = KL_distance(mfcc_mean_1, mfcc_mean_0, mfcc_covar_1, mfcc_covar_0, mfcc_inv_covar_0, dim)
	return 0.5 * first_distance + 0.5 * second_distance


# def SKL_distance(mfcc_mean_0, mfcc_mean_1, mfcc_covar_0, mfcc_covar_1, mfcc_inv_covar_0, mfcc_inv_covar_1, dim = 13):
# 	first_adder = mfcc_inv_covar_0.dot(mfcc_covar_1)
# 	first_adder = trace(first_adder)
# 	second_adder = mfcc_inv_covar_1.dot(mfcc_covar_0)
# 	second_adder = trace(second_adder)
# 	mean_difference = subtract(mfcc_mean_1, mfcc_mean_0)
# 	third_adder = ((mfcc_inv_covar_0 + mfcc_inv_covar_1).dot(mean_difference)).dot(mean_difference.T)
# 	# third_adder = trace(third_adder)
# 	result = first_adder + second_adder + third_adder - 2 * dim
# 	# print result
# 	return result


def KL_distance(mfcc_mean_0, mfcc_mean_1, mfcc_covar_0, mfcc_covar_1, mfcc_inv_covar_1, dim = 13):
	"""
	Computes (non-symmetric) Kullback-Leibler divergence
	"""

	first_adder = mfcc_inv_covar_1.dot(mfcc_covar_0)
	first_adder = trace(first_adder)
	mean_difference = subtract(mfcc_mean_1, mfcc_mean_0)
	second_adder = ((mean_difference.T).dot(mfcc_inv_covar_1)).dot(mean_difference)
	last_adder = log(det(mfcc_covar_0) / det(mfcc_covar_1))
	total_distance = 0.5 * (first_adder + second_adder - dim - last_adder)
	return total_distance


def matrix_sum(matrix_a, matrix_b):
	output_matrix = []
	for row_index in range(len(matrix_a)):
		tmp_vector = []
		row_a = matrix_a[row_index]
		row_b = matrix_b[row_index] 
		for column_index in range(len(row_a)):
			tmp_sum = row_a[column_index] + row_b[column_index]
			tmp_vector.append(tmp_sum)
		output_matrix.append(tmp_vector)
	return output_matrix


def build_diag_cov_matrix_from_var_vector(var_vector):
	"""
	Builds a diagonal covariance matrix starting from the vector of variance.
	"""

	diag_cov_matrix = []
	for i in range(13):
		tmp_vector = []
		for j in range(13):
			if (i == j):
				tmp_vector.append(var_vector[i])
			else:
				tmp_vector.append(0)
		diag_cov_matrix.append(tmp_vector)	
	return diag_cov_matrix


def build_matrix_from_vector(input_vector, dim):
	"""
	Builds a 13x13 full covariance matrix starting from a vector of 169 dimensions.
	"""

	covar_matrix = []
	for i in range(13):
		tmp_vector = []
		for j in range(13):
			tmp_vector.append(input_vector[i * 13 + j])
		covar_matrix.append(tmp_vector)
	covar_matrix = asarray(covar_matrix)
	covar_matrix.reshape((dim, dim))
	return covar_matrix


def key_distance(src_key, tgt_key):
	"""
	Returns true if src and target key modes are compatible (i.e. are not dissonant). 
	"""

	traspose_value = 0
	suitable_values = [1, 6, 10, 11, 15, 20]
	suitable_values_traslated = []
	minorMode = False
	if src_key in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]: # minor mode
		minorMode = True
		src_key += 5
		if src_key > 24:
			src_key = src_key % 24
	
	if tgt_key in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]: # minor mode
		minorMode = True
		tgt_key += 5
		if tgt_key > 24:
			tgt_key = tgt_key % 24

	distance = max(src_key, tgt_key) - min(src_key, tgt_key) + 1
	return distance in suitable_values


def bpm_distance(src_bpm, tgt_bpm):
	"""
	Computes the distance between two input bmp values, in a way that eventually forgives wheter some value has not been computed properly (e.g. you get 140bpm instead of 70)
	"""
	
	numerator = float(max(src_bpm, tgt_bpm))
	denominator = float(min(src_bpm, tgt_bpm))
	ratio = numerator/denominator
	min_distance = 30 * abs(ratio - 1)
	for i in [2, 4, 6]:
		distance = 30 * abs(ratio - i)
		if distance < min_distance:
			min_distance = distance
	return min_distance


def scan_needed(music_dir, scan_output, lines_in_file):
	""" 
	Returns true if I need to scan music library to check for similarities between songs.
	"""

	files = get_json_files_in_path(music_dir)

	# Perform a check on the number of files with descriptors and the number of files scanned with musly/custom scanner 
	output = (len(files) != len(scan_output.all_songs)) or (lines_in_file < len(files) + 2)
	return output 

	
def load_song_descriptors(filename):
	"""
	Load beats descriptors for a song, as well as song acousticness.
	"""

	# we don't need to check if 'filename' has descriptors, we've already done that in the main method
	input_filename = ('.').join(filename.split('.')[:-1]) + '.json'

	with open(input_filename) as data_file:
		json_content = json.load(data_file)

	return json_content["song.acousticness"], json_content["song.title"], json_content["song.artist"], \
			json_content["song.year"], json_content["beats.descriptors"]


def get_json_files_in_path(path):
	files = []
	for dirpath, dirnames, filenames in os.walk(path):
		for filename in [f for f in filenames if f.endswith(".json")]:
			files.append(os.path.join(dirpath, filename))
	return files


def echonest_formatter(track):
	"""
	"""

	analysis = {}

	meta = {}

	if 'id' in dir(track): 
		meta['en_id'] = track.id

	if 'song_id' in dir(track):  
		meta['en_song_id'] = track.song_id

	if 'artist_id' in dir(track):
		meta['en_artist_id'] = track.artist_id

	if 'meta' in dir(track):
		if 'title' in track.meta.keys():
			meta['en_title'] = track.meta['title']
		if 'artist' in track.meta.keys():
			meta['en_artist'] = track.meta['artist']
		if 'album' in track.meta.keys():
			meta['en_album'] = track.meta['album']
		if 'seconds' in track.meta.keys():
			meta['en_duration_seconds'] = track.meta['seconds']
		if 'analyzer_version' in track.meta.keys():
			meta['en_analyzer_version'] = track.meta['analyzer_version']
		if 'sample_rate' in track.meta.keys():
			meta['en_sample_rate'] = track.meta['sample_rate']
		if 'timestamp' in track.meta.keys():
			meta['en_timestamp'] = track.meta['timestamp']
		if 'genre' in track.meta.keys():
			meta['en_genre'] = track.meta['genre']
		if 'bitrate' in track.meta.keys():
			meta['en_bitrate'] = track.meta['bitrate']
		if 'analysis_time' in track.meta.keys():
			meta['en_analysis_time'] = track.meta['analysis_time']
	else:
		if 'title' in dir(track):
			meta['en_title'] = track.title
		if 'artist' in dir(artist):
			meta['en_artist'] = track.artist
		if 'bitrate' in dir(track):
			meta['en_bitrate'] = track.bitrate
		if 'analyzer_version' in dir(track):
			meta['en_analyzer_version'] = track.analyzer_version

	if 'audio_md5' in dir(track):
		meta['en_audio_md5'] = track.audio_md5
	if 'sample_md5' in dir(track):
		meta['en_sample_md5'] = track.sample_md5
	if 'md5' in dir(track):
		meta['en_md5'] = track.md5
	if 'decoder' in dir(track):
		meta['en_decoder'] = track.decoder
	if 'decoder_version' in dir(track):
		meta['en_decoder_version'] = track.decoder_version
	if 'analysis_channels' in dir(track):
		meta['en_analysis_channels'] = track.analysis_channels
	if 'analysis_sample_rate' in dir(track):
		meta['en_analysis_sample_rate'] = track.analysis_sample_rate
	if 'num_samples' in dir(track):
		meta['en_num_samples'] = track.num_samples
	if 'samplerate' in dir(track):
		meta['en_sample_rate'] = track.samplerate
	if 'window_seconds' in dir(track):
		meta['en_window_seconds'] = track.window_seconds
	if 'duration' in dir(track):
		meta['en_duration'] = track.duration

	analysis['meta'] = meta

	desc = {}

	if 'energy' in dir(track):
		desc['en_energy'] = track.energy
	if 'loudness' in dir(track):
		desc['en_loudness'] = track.loudness
	if 'end_of_fade_in' in dir(track):
		desc['en_end_of_fade_in'] = track.end_of_fade_in
	if 'start_of_fade_out' in dir(track):
		desc['en_start_of_fade_out'] = track.start_of_fade_out

	if 'acousticness' in dir(track):
		desc['en_acousticness'] = track.acousticness
	if 'danceability' in dir(track):
		desc['en_danceability'] = track.danceability
	if 'liveness' in dir(track):
		desc['en_liveness'] = track.liveness
	if 'speechiness' in dir(track):
		desc['en_speechiness'] = track.speechiness
	if 'valence' in dir(track):
		desc['en_valence'] = track.valence

	if 'key' in dir(track):
		desc['en_key'] = track.key
		if 'key_confidence' in dir(track):
			desc['en_key_confidence'] = track.key_confidence
	if 'mode' in dir(track):
		desc['en_mode'] = track.mode
		if 'mode_confidence' in dir(track):
			desc['en_mode_confidence'] = track.mode_confidence

	if 'time_signature' in dir(track):
		desc['en_time_signature'] = track.time_signature
		if 'time_signature_confidence' in dir(track):
			desc['en_time_signature_confidence'] = \
				track.time_signature_confidence
	if 'tempo' in dir(track):
		desc['en_tempo'] = track.tempo
		if 'tempo_confidence' in dir(track):
			desc['en_tempo_confidence'] = track.tempo_confidence

	analysis['desc'] = desc

	rhythm = {}

	bars = []

	if 'bars' in dir(track):
		for bar in track.bars:
			b = {}
			if 'start' in bar.keys():
				b['en_start'] = bar['start']
			if 'duration' in bar.keys():
				b['en_duration'] = bar['duration']
			if 'confidence' in bar.keys():
				b['en_confidence'] = bar['confidence']
			bars.append(b)

	rhythm['bars'] = bars

	beats = []

	if 'beats' in dir(track):
		for beat in track.beats:
			b = {}
			if 'start' in beat.keys():
				b['en_start'] = beat['start']
			if 'duration' in beat.keys():
				b['en_duration'] = beat['duration']
			if 'confidence' in beat.keys():
				b['en_confidence'] = beat['confidence']
			beats.append(b)

	rhythm['beats'] = beats

	tatums = []

	if 'tatums' in dir(track):
		for tatum in track.tatums:
			t = {}
			if 'start' in tatum.keys():
				t['en_start'] = tatum['start']
			if 'duration' in tatum.keys():
				t['en_duration'] = tatum['duration']
			if 'confidence' in tatum.keys():
				t['en_confidence'] = tatum['confidence']
			tatums.append(t)

	rhythm['tatums'] = tatums   

	analysis['rhythm'] = rhythm

	structure = {}

	sections = []

	if 'sections' in dir(track):
		for section in track.sections:
			s = {}
			if 'start' in section.keys():
				s['en_start'] = section['start']
			if 'duration' in section.keys():
				s['en_duration'] = section['duration']
			if 'confidence' in section.keys():
				s['en_confidence'] = section['confidence']
			if 'loudness' in section.keys():
				s['en_loudness'] = section['loudness']
			if 'key' in section.keys():
				s['en_key'] = section['key']
				if 'key_confidence' in section.keys():
					s['en_key_confidence'] = \
						section['key_confidence']
			if 'mode' in section.keys():
				s['en_mode'] = section['mode']
				if 'mode_confidence' in section.keys():
					s['en_mode_confidence'] = \
						section['mode_confidence']
			if 'time_signature' in section.keys():
				s['en_time_signature'] = section['time_signature']
				if 'time_signature_confidence' in section.keys():
					s['en_time_signature_confidence'] = \
						section['time_signature_confidence']
			if 'tempo' in section.keys():
				s['en_tempo'] = section['tempo']
				if 'tempo_confidence' in section.keys():
					s['en_tempo_confidence'] = \
						section['tempo_confidence']
			sections.append(s)

	structure['sections'] = sections

	segments = []

	if 'segments' in dir(track):
		for segment in track.segments:
			s = {}
			if 'start' in segment.keys():
				s['en_start'] = segment['start']
			if 'duration' in segment.keys():
				s['en_duration'] = segment['duration']
			if 'confidence' in segment.keys():
				s['en_confidence'] = segment['confidence']
			if 'loudness_start' in segment.keys():
				s['en_loudness_start'] = segment['loudness_start']
			if 'loudness_max' in segment.keys():
				s['en_loudness_max'] = segment['loudness_max']
			if 'loudness_max_time' in segment.keys():
				s['en_loudness_max_time'] = segment['loudness_max_time']
			if 'timbre' in segment.keys():
				for i in range(len(segment['timbre'])):
					if (i + 1) < 10:
						i_str = '0' + str(i + 1)
					else:
						i_str = str(i + 1)
					s['en_timbre_' + i_str] = segment['timbre'][i]
			if 'pitches' in segment.keys():
				for i in range(len(segment['pitches'])):
					if (i + 1) < 10:
						i_str = '0' + str(i + 1)
					else:
						i_str = str(i + 1)
					s['en_pitch_' + i_str] = segment['pitches'][i]
			
			segments.append(s)

	structure['segments'] = segments
	analysis['structure'] = structure
	
	strings = {}

	if 'codestring' in dir(track):
		strings['en_code'] = track.codestring
		if 'code_version' in dir(track):
			strings['en_code_version'] = track.code_version
	if 'echoprintstring' in dir(track):
		strings['en_echoprint'] = track.echoprintstring
		if 'echoprint_version' in dir(track):
			strings['en_echoprint_version'] = track.echoprint_version
	if 'rhythmstring' in dir(track):
		strings['en_rhythm'] = track.rhythmstring
		if 'rhythm_version' in dir(track):
			strings['en_rhythm_version'] = track.rhythm_version
	if 'synchstring' in dir(track):
		strings['en_synch'] = track.synchstring
		if 'synch_version' in dir(track):
			strings['en_synch_version'] = track.synch_version

	analysis['strings'] = strings

	return analysis



if __name__ == "__main__":
	tmp_vector = []
	for i in range(13):
		tmp_vector.append(i + 1)
	diag_matrix = build_diag_cov_matrix_from_var_vector(tmp_vector)
	print diag_matrix
	inv_matrix = build_inv_of_diag_matrix(diag_matrix)
	print inv_matrix
