from ConfigParser import SafeConfigParser
import pyechonest.config as echo_conf
import pyechonest.track as echo_track
import os
import sys
import json
from util import hasJsonDescriptor, echonest_formatter, get_json_files_in_path
import numpy
from numpy.linalg import inv, det
from socket import error as SocketError
import errno
import time
import essentia
from essentia.standard import Extractor, MonoLoader, Trimmer, Mean, FrameGenerator, Spectrum, SpectralPeaks, Dissonance, BarkBands, Windowing, \
	ZeroCrossingRate, OddToEvenHarmonicEnergyRatio, EnergyBand, MetadataReader, OnsetDetection, Onsets, CartesianToPolar, FFT, MFCC, SingleGaussian
from build_map import build_map

sampleRate = 44100
frameSize = 2048
hopSize = 1024
windowType = "hann"

mean = Mean()

keyDetector = essentia.standard.Key(pcpSize = 12)
spectrum = Spectrum()
window = Windowing(size = frameSize, zeroPadding = 0, type = windowType)
mfcc = MFCC()
gaussian = SingleGaussian()
od = OnsetDetection(method = 'hfc')
fft = FFT() # this gives us a complex FFT
c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)
onsets = Onsets(alpha = 1)

# dissonance
spectralPeaks = SpectralPeaks(sampleRate = sampleRate, orderBy='frequency')
dissonance = Dissonance()

# barkbands
barkbands = BarkBands(sampleRate = sampleRate)

# zero crossing rate
# zerocrossingrate = ZeroCrossingRate()

# odd-to-even harmonic energy ratio
# odd2evenharmonicenergyratio = OddToEvenHarmonicEnergyRatio()

# energybands
# energyband_bass = EnergyBand(startCutoffFrequency = 20.0, stopCutoffFrequency = 150.0, sampleRate = sampleRate)
# energyband_high = EnergyBand(startCutoffFrequency = 4000.0, stopCutoffFrequency = 20000.0, sampleRate = sampleRate)


parser = SafeConfigParser()
parser.read('config.ini')

api_key = parser.get('echonest', 'apikey')
echo_conf.ECHO_NEST_API_KEY = api_key

def compute_folder_descriptors(path):
	"""
	Scan the folder and look for files that don't have descriptors.
	For each of them, call the method for computing descriptors. 
	"""

	files_to_scan = []

	# find files that need to be analyzed
	for dirpath, dirnames, filenames in os.walk(path):
		for filename in [f for f in filenames if f.endswith(".mp3")]:
			if not hasJsonDescriptor(os.path.join(dirpath, filename)):
				files_to_scan.append(os.path.join(dirpath, filename))

	if (files_to_scan):
		files_to_scan.sort()
		
		# Compute descriptors for files not analyzed yet
		for index, file_ in enumerate((files_to_scan)[:]):
			counter = 0
			while True:
				try:
					analysis = perform_echonest_analysis(file_)
					data = select_features(analysis, file_)
					output_file_path = ('.').join(file_.split('.')[:-1]) + ".json"
					with open(output_file_path, 'w') as outfile:
						json.dump(data, outfile)
						# output_json = json.dumps(data)
						# compressed_data = zlib.compress(output_json)
						# outfile.write(compressed_data)
					print "Descriptors computed for " + str(index + 1) + " out of " + str(len(files_to_scan)) + " files."
				except:
					counter += 1
					tries = 2
					if counter != tries:
						print "There was a problem. Retrying soon..."
					elif counter == tries:
						print "Couldn't compute descriptors for file " + file_
						break
					time.sleep(10)
					continue
				break

		print "All files analyzed!"
		return

	else:
		print "No files to analyze"
		return


def select_features(analysis, filename):
	"""
	Select only interesting features from echonest analysis output.
	Compute additional descriptors with essentia.
	"""

	title, artist, year, danceability, speechiness, valence, acousticness, energy = get_song_features(analysis, filename)

	mfccs_coeff_song = []
	dict_keys = ('beats.descriptors', 'song.mfcc.mean')
	song_features = dict.fromkeys(dict_keys)
	all_bars_descriptors = []

	audio_file = MonoLoader(filename = filename, sampleRate = sampleRate)()
	bar_counter = 0

	# compute descriptors for each bar
	for bar_index in range(len(analysis["rhythm"]["bars"])):
		bar_dict_keys = ('mfcc.mean', 'mfcc.covar', 'mfcc.invcovar', 'chroma.first', 'chroma.fourth', 'barks', 'others')
		bar_features = dict.fromkeys(bar_dict_keys)
		general_descriptors = []
		actual_bar = analysis["rhythm"]["bars"][bar_index]
		actual_bar_beg = actual_bar['en_start']
		actual_bar_end = actual_bar['en_start'] + actual_bar['en_duration']

		general_descriptors.append(actual_bar_beg)
		general_descriptors.append(actual_bar_end)
		trimmer = Trimmer(startTime = actual_bar_beg, endTime = actual_bar_end)
		audio_segment = trimmer(audio_file)

		mfccs_bar, bark_vector, onset_rate, bar_dissonance = compute_essentia_descriptors(audio_segment, actual_bar_beg, actual_bar_end)
		for val in mfccs_bar:
			mfccs_coeff_song.append(val)
		mfcc_mean_bar, mfcc_covar_bar, mfcc_invcovar_bar = get_gaussian_from_mfcc(mfccs_bar)
		if not mfcc_mean_bar:
			continue  # skip segments for whom we couldn't compute a gaussian model (it allows us to save some disk space)

		mfcc_mean = []
		mfcc_cov = []
		mfcc_incov = []

		for val in mfcc_mean_bar:
			mfcc_mean.append(float(val))
		for row in mfcc_covar_bar:
			for val in row:
				mfcc_cov.append(float(val))
		for row in mfcc_invcovar_bar:
			for val in row:
				mfcc_incov.append(float(val))

		bar_features['mfcc.mean'] = mfcc_mean  # before it was: ... = mfcc_mean_bar
		bar_features['mfcc.covar'] = mfcc_cov  # before it was: ... = mfcc_covar_bar
		bar_features['mfcc.invcovar'] = mfcc_incov
		bar_features['barks'] = bark_vector

		counter = 0
		for segment in analysis['structure']['segments']:
			segment_beg = segment['en_start']
			segment_end = segment['en_start'] + segment['en_duration']
			if sectionIsInBar(segment_beg, segment_end, actual_bar_beg, actual_bar_end):
				counter += 1
				if counter == 1:
					pitch_vector = compute_pitch_vector(segment)
					bar_features['chroma.first'] = pitch_vector

				if greaterOrEqual(segment_end, actual_bar_end): # we are in the fourth beat of the bar
					pitch_vector = compute_pitch_vector(segment)
					bar_features['chroma.fourth'] = pitch_vector
					break

		bar_bpm = []
		bar_loudness = []
		for section in analysis['structure']['sections']:
			section_beg = section['en_start']
			section_end = section['en_start'] + section['en_duration']
			if sectionIsInBar(section_beg, section_end, actual_bar_beg, actual_bar_end):
				section_weight = (min(section_end, actual_bar_end) - max(section_beg, actual_bar_beg)) / (actual_bar_end - actual_bar_beg)
				try:
					bar_bpm.append(section['en_tempo'] * section_weight)
				except:
					# BPM value not found, use the BPM value of the song.
					bar_bpm.append(analysis['desc']['en_tempo'] * section_weight)
				try:
					bar_loudness.append(section['en_loudness'] * section_weight)
				except:
					# Loudness value not found, use the loudness value of the song.
					bar_bpm.append(analysis['desc']['en_loudness'] * section_weight)
				if greaterOrEqual(section_end, actual_bar_end):
					general_descriptors.append(sum(bar_bpm))
					general_descriptors.append(sum(bar_loudness))
					general_descriptors.append(bar_dissonance)
					# general_descriptors.append(mean(pool["zerocrossingrate"]))
					general_descriptors.append(onset_rate)
					break

		bar_features['others'] = general_descriptors
		all_bars_descriptors.append(bar_features)
		bar_counter += 1
		del audio_segment, trimmer

	mfcc_mean_song, mfcc_cov_song, mfcc_inv_cov_song = get_gaussian_from_mfcc(mfccs_coeff_song)
	song_features['song.mfcc.mean'] = mfcc_mean_song
	song_features['song.mfcc.covar'] = mfcc_cov_song
	song_features['song.mfcc.invcovar'] = mfcc_inv_cov_song	
	song_features['song.danceability'] = danceability
	song_features['song.speechiness'] = speechiness
	song_features['song.valence'] = valence
	song_features['song.energy'] = energy  # you may want to delete everything until here
	song_features['beats.descriptors'] = all_bars_descriptors			
	song_features['song.acousticness'] = acousticness
	song_features['song.title'] = title
	song_features['song.artist'] = artist
	song_features['song.year'] = year
	del audio_file
	return song_features


def perform_echonest_analysis(filename):
	"""
	Perform the analysis of an audio file with echonest.
	"""
	analysis = {}
	print 'Computing echonest\'s descritors for ' + filename + "..."
	pytrack = echo_track.track_from_filename(filename)
	pytrack.get_analysis()
	analysis = echonest_formatter(pytrack)
	print 'Done'
	return analysis


def sectionIsInBar(sec_b, sec_e, bar_b, bar_e):
	return (greaterOrEqual(bar_b, sec_b) and lessOrEqual(bar_e, sec_e)) or \
			(lessOrEqual(bar_b, sec_b) and greaterOrEqual(bar_e, sec_e)) or \
			(lessOrEqual(bar_b, sec_b) and lessOrEqual(bar_e, sec_e) and bar_e > sec_b) or \
			(greaterOrEqual(bar_b, sec_b) and greaterOrEqual(bar_e, sec_e) and bar_b < sec_e)


def lessOrEqual(a, b): 
	"""
	Checks if a <= b.
	"""
	return (a < b) or (abs(a - b) < 0.0001)


def greaterOrEqual(a, b): 
	"""
	Checks if a >= b.
	"""
	return (a > b) or (abs(a - b) < 0.0001)


def compute_pitch_vector(segment):
	"""
	Returns a 13-D vector. First 12 values correspond to HPCP, the 13th indicates the mode.
	"""
	pitch_vector = []
	for str_ in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
		str_id = 'en_pitch_' + str_
		pitch_vector.append(segment[str_id])
	traslated_chroma = []
	for val in pitch_vector[9:]:
		traslated_chroma.append(val)
	for val in pitch_vector[:9]:
		traslated_chroma.append(val)
	key, scale, strength, firstToSecondRelativeStrength = keyDetector(traslated_chroma)
	key_mode_val = get_key_mode_val(key, scale)
	pitch_vector.append(key_mode_val)
	return pitch_vector


def get_key_mode_val(key, scale):
	"""
	Returns a value that univocally indicates the mode. 
	E.g.: C+ is 1, C- is 2, C#+ is 3, C#- is 4, D+ is 5, ...
	"""
	key_mode_val = 0
	if "C" in key:
		key_mode_val += 1
	elif "D" in key:
		key_mode_val += 5
	elif "E" in key:
		key_mode_val += 9
	elif "F" in key:
		key_mode_val += 11
	elif "G" in key:
		key_mode_val += 15
	elif "A" in key:
		key_mode_val += 19
	elif "B" in key:
		key_mode_val += 23
	if "#" in key:
		key_mode_val += 2
	if scale == "minor":
		key_mode_val += 1
	return key_mode_val


def get_song_features(analysis, filename):
	metadata_reader = MetadataReader(filename = filename) 
	metadata = metadata_reader()
	track_title = metadata[0]
	track_artist = metadata[1]
	track_year = metadata[6]
	danceability = analysis['desc']['en_danceability']
	speechiness = analysis['desc']['en_speechiness']
	valence = analysis['desc']['en_valence']
	acousticness = analysis['desc']['en_acousticness']
	energy = analysis['desc']['en_energy']
	return track_title, track_artist, track_year, danceability, speechiness, valence, acousticness, energy


def get_gaussian_from_mfcc(mfccs):
	"""
	Returns mean vector, covariance and inverse covariance matrices of the single gaussian model representing the given mfcc values. 
	"""

	mfcc_mean_vec = []
	mfcc_covar_float = []
	mfcc_invcovar_float = []
	try:
		mfcc_mean, mfcc_cov, mfcc_inv_cov = gaussian(mfccs)
		det_mfcc_cov = det(mfcc_cov)
		if instable_inverse_covariance_matrix(mfcc_inv_cov) or (det_mfcc_cov < 0):
			# print "Warning: found an instable covariance matrix. Using diagonal matrix instead."
			for i in range(len(mfcc_cov)):
				tmp_vector = []
				for j in range(len(mfcc_cov[i])):
					if i == j:
						tmp_vector.append(float(mfcc_cov[i][j]))
					else:
						tmp_vector.append(float(0))
				mfcc_covar_float.append(tmp_vector)
			inv_matrix = inv(mfcc_covar_float)
			for row in inv_matrix:
				tmp_vector = []
				for val in row:
					tmp_vector.append(float(val))
				mfcc_invcovar_float.append(tmp_vector)	
		else:
			for row in mfcc_cov:
				tmp_vector = []
				for val in row:
					tmp_vector.append(float(val))
				mfcc_covar_float.append(tmp_vector)
			for row in mfcc_inv_cov:
				tmp_vector = []
				for val in row:
					tmp_vector.append(float(val))
				mfcc_invcovar_float.append(tmp_vector)
		for val in mfcc_mean:
			mfcc_mean_vec.append(float(val))
	except:
		# Matrix could be singular, therefore we couldn't be able to compute a gaussian model
		pass
	return mfcc_mean_vec, mfcc_covar_float, mfcc_invcovar_float


def instable_inverse_covariance_matrix(input_matrix):
	for row in input_matrix:
		for val in row:
			if abs(val) > 1:
				return True
	return False


def compute_essentia_descriptors(audio_segment, actual_bar_beg, actual_bar_end):
	"""
	Computes the values of selected descriptors in the given audio segment.
	"""
	frames = FrameGenerator(audio_segment, frameSize = frameSize, hopSize = hopSize)
	mfccs_bar = []
	bark_vector = [0] * 27  
	pool = essentia.Pool()
	total_frames = frames.num_frames()

	for frame in frames:
		frame_windowed = window(frame)
		frame_spectrum = spectrum(frame_windowed)
		(frame_frequencies, frame_magnitudes) = spectralPeaks(frame_spectrum)
		mag, phase, = c2p(fft(frame_windowed))
		pool.add('onsets.hfc', od(mag, phase))
		frame_dissonance = dissonance(frame_frequencies, frame_magnitudes)
		pool.add('dissonance', frame_dissonance)
		# pool.add('zerocrossingrate', zerocrossingrate(frame))
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(window(frame)))
		mfccs_bar.append(mfcc_coeffs)
		frame_barkbands = barkbands(frame_spectrum)
		for i in range(27):
			bark_vector[i] += frame_barkbands[i] / total_frames

	onsets_hfc = onsets(essentia.array([ pool['onsets.hfc'] ]), [ 1 ])
	onset_rate = float(len(onsets_hfc))/(actual_bar_end - actual_bar_beg)
	bar_dissonance = mean(pool["dissonance"])

	return mfccs_bar, bark_vector, onset_rate, bar_dissonance


# def compute_global_values(path):
# 	"""
# 	Update global values: lowest and highest, 1st, 2nd and 3rd quantile for each feature that will be controlled through a slider in the GUI.
# 	"""

# 	print "Updating global values..."

# 	# find json files that we'll scan
# 	files = get_json_files_in_path(path)
# 	all_loudnesses, all_dissonances, all_onsetsrates, all_barks, all_acousticnesses = collect_values_in_collection(files)

# 	arrays_len = len(all_loudnesses) # this amount equals len(all_dissonances) and len(all_onsetsrates)
# 	acstc_len = len(all_acousticnesses)

# 	first_quantile_index = arrays_len/4
# 	second_quantile_index = arrays_len/2
# 	third_quantile_index = arrays_len * 3/4

# 	# [lowest, 1st quantile, 2nd quantile, 3rd quantile, highest]
# 	loudness_values_vector = [all_loudnesses[0], all_loudnesses[first_quantile_index], all_loudnesses[second_quantile_index], \
# 								all_loudnesses[third_quantile_index], all_loudnesses[arrays_len -1]]
# 	dissonance_values_vector = [all_dissonances[0], all_dissonances[first_quantile_index], all_dissonances[second_quantile_index], \
# 								all_dissonances[third_quantile_index], all_dissonances[arrays_len -1]]
# 	onsetsrates_values_vector = [all_onsetsrates[0], all_onsetsrates[first_quantile_index], all_onsetsrates[second_quantile_index], \
# 								all_onsetsrates[third_quantile_index], all_onsetsrates[arrays_len -1]]
# 	acousticness_values_vector = [all_acousticnesses[0], all_acousticnesses[acstc_len/4], all_acousticnesses[acstc_len/2], \
# 								all_acousticnesses[acstc_len *3/4], all_acousticnesses[acstc_len -1]]
# 	# for the bark values, we store the first tertile for each band
# 	barks_values_vector = []
# 	for index in range(27):
# 		tmp_vector = []
# 		for i in range(arrays_len):
# 			tmp_vector.append(all_barks[i][index])
# 		barks_values_vector.append(tmp_vector[arrays_len/3])

# 	sliders_values = {}
# 	sliders_values.update({"loudness": loudness_values_vector})
# 	sliders_values.update({"dissonance": dissonance_values_vector})
# 	sliders_values.update({"onsets": onsetsrates_values_vector})
# 	sliders_values.update({"acousticness": acousticness_values_vector})
# 	sliders_values.update({"bark": barks_values_vector})

# 	sliders_output_file_path = path + "sliders_values"	
# 	with open(sliders_output_file_path, 'w') as outfile:
# 		json.dump(sliders_values, outfile)

# 	print "Done!"


# def collect_values_in_collection(files):
# 	all_loudnesses = []
# 	all_dissonances = []
# 	all_onsetsrates = []
# 	all_barks = []
# 	all_acousticnesses = []

# 	for file_ in files:
# 		with open(file_) as data_file:
# 			json_content = json.load(data_file)

# 		if json_content["song.acousticness"]: 
# 			all_acousticnesses.append(json_content["song.acousticness"])
# 		else: # acousticness could be 'None'
# 			all_acousticnesses.append(0)
# 		for beat in json_content["beats.descriptors"]:
# 			try:
# 				all_loudnesses.append(beat["others"][3])
# 				all_dissonances.append(beat["others"][4])
# 				all_onsetsrates.append(beat["others"][5])
# 				all_barks.append(beat["barks"])
# 			except:
# 				continue
# 	all_loudnesses = sorted(all_loudnesses)
# 	all_dissonances = sorted(all_dissonances)
# 	all_onsetsrates = sorted(all_onsetsrates)
# 	all_acousticnesses = sorted(all_acousticnesses)
# 	return all_loudnesses, all_dissonances, all_onsetsrates, all_barks, all_acousticnesses


if __name__ == "__main__":
	try:
		if len(sys.argv) < 2:
			print "Usage: " + sys.argv[0] + " <path_to_music_folder>"
			sys.exit(-1)
		music_dir = sys.argv[1]
		if music_dir[-1] != "/":
			music_dir += "/"
		compute_folder_descriptors(music_dir)
		#compute_global_values(music_dir)
		build_map(music_dir)
	except KeyboardInterrupt:
		sys.exit(-1)

