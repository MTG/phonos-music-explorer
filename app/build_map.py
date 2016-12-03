import os.path
import sys 
from util import SKL_distance, get_json_files_in_path, build_matrix_from_vector, bpm_distance
import json
import random
from numpy.linalg import inv, det, norm
from operator import itemgetter
from numpy import sqrt, reshape, asarray
import time

dimensions = 20
pivots_percentage = 0
filter_size = 0.1

def build_map(music_dir):
	sys.stdout.write('Collecting all descriptors in folder...')
	sys.stdout.flush()
	files = get_json_files_in_path(music_dir)
	beats, all_loudnesses, all_dissonances, all_onsetsrates, all_barks, all_acousticnesses = collect_all_beats_descriptors(files)
	sys.stdout.write('Done\n')
	sys.stdout.flush()
	print "Segments found:", len(beats)
	pivots_indexes = get_random_pivots(beats)
	print 'Finding pivots, completion: '
	all_pivots = find_all_corresponding_pivots(beats, pivots_indexes)
	print "Done"
	sys.stdout.write('Computing points positions in new space, completion:\n')
	sys.stdout.flush()
	k_dimensions_map = compute_points_positions(beats, all_pivots)
	sys.stdout.write('Done\n')
	sys.stdout.write('Writing map to output file...')
	write_sliders_values(music_dir, all_loudnesses, all_dissonances, all_onsetsrates, all_barks, all_acousticnesses)
	write_to_output_file(beats, k_dimensions_map, music_dir)
	sys.stdout.write('Done\n')
	sys.stdout.flush()


def compute_points_positions(beats, all_pivots):
	map_of_points = []
	map_building_completion = 0
	for index, beat in enumerate(beats):
		point = []
		for j in range(dimensions):
			pivots_in_j = all_pivots[j]
			pivot_j1_index = pivots_in_j["first"]
			pivot_j2_index = pivots_in_j["second"]
			pivot_j1 = beats[pivot_j1_index]
			pivot_j2 = beats[pivot_j2_index]
			distance_j1_j2 = pivots_in_j["distance"]
			if pivot_j1_index != index:
				first_adder = get_distance(beat, pivot_j1) ** 2
			else:
				first_adder = 0
			second_adder = distance_j1_j2 ** 2
			if pivot_j2_index != index:
				third_adder = get_distance(beat, pivot_j2) ** 2
			else:
				third_adder = 0
			point_coord_j = (first_adder + second_adder - third_adder) / (2 * distance_j1_j2)
			point.append(point_coord_j)
		map_of_points.append(point)
		map_building_completion += 1./len(beats) * 100
		status(map_building_completion)
	return map_of_points


def	write_sliders_values(path, all_loudnesses, all_dissonances, all_onsetsrates, all_barks, all_acousticnesses):
	print "Writing sliders values..."
	arrays_len = len(all_loudnesses) # this amount equals len(all_dissonances) and len(all_onsetsrates)
	acstc_len = len(all_acousticnesses)

	first_quantile_index = arrays_len/4
	second_quantile_index = arrays_len/2
	third_quantile_index = arrays_len * 3/4

	# [lowest, 1st quantile, 2nd quantile, 3rd quantile, highest]
	loudness_values_vector = [all_loudnesses[0], all_loudnesses[first_quantile_index], all_loudnesses[second_quantile_index], \
								all_loudnesses[third_quantile_index], all_loudnesses[arrays_len -1]]
	dissonance_values_vector = [all_dissonances[0], all_dissonances[first_quantile_index], all_dissonances[second_quantile_index], \
								all_dissonances[third_quantile_index], all_dissonances[arrays_len -1]]
	onsetsrates_values_vector = [all_onsetsrates[0], all_onsetsrates[first_quantile_index], all_onsetsrates[second_quantile_index], \
								all_onsetsrates[third_quantile_index], all_onsetsrates[arrays_len -1]]
	acousticness_values_vector = [all_acousticnesses[0], all_acousticnesses[acstc_len/4], all_acousticnesses[acstc_len/2], \
								all_acousticnesses[acstc_len *3/4], all_acousticnesses[acstc_len -1]]
	# for the bark values, we store the first tertile for each band
	barks_values_vector = []
	for index in range(27):
		tmp_vector = []
		for i in range(arrays_len):
			tmp_vector.append(all_barks[i][index])
		barks_values_vector.append(tmp_vector[arrays_len/3])

	sliders_values = {}
	sliders_values.update({"loudness": loudness_values_vector})
	sliders_values.update({"dissonance": dissonance_values_vector})
	sliders_values.update({"onsets": onsetsrates_values_vector})
	sliders_values.update({"acousticness": acousticness_values_vector})
	sliders_values.update({"bark": barks_values_vector})

	sliders_output_file_path = path + "sliders_values"	
	with open(sliders_output_file_path, 'w') as outfile:
		json.dump(sliders_values, outfile)



def find_nn_for_each_point(beats, k_dimensions_map):
	all_similarities = []
	number_of_neighbours = int(len(k_dimensions_map) * filter_size)
	filter_percentage = 0
	for index, point in enumerate(k_dimensions_map):
		point = asarray(point)
		n_neighbors = get_euclidean_neighbors(point, beats, k_dimensions_map, number_of_neighbours, index)
		# n_neighbors = refine(index, beats, n_neighbors)
		all_similarities.append(n_neighbors)
		filter_percentage += 1./len(k_dimensions_map) * 100
		status(filter_percentage)
	return all_similarities


def write_to_output_file(beats, k_dimensions_map, music_dir):
	with open(music_dir + "/sliders_values") as data_file:
		json_content = json.load(data_file)
	bark_thresholds_vector = json_content["bark"]
	output_file_path = music_dir + "outputmap"
	output_dict = {}
	all_beats = []
	_70s_list = []
	_80s_list = []
	_90s_list = []
	_00s_list = []
	_10s_list = []
	for index, beat in enumerate(beats):
		point_coords = k_dimensions_map[index]
		beat_dict = {}
		# beat_dict.update({"id": index})
		beat_bark = beat["barks"]
		bark_bands_over_thresh = find_bands_over_thresh(beat_bark, bark_thresholds_vector)
		beat_dict.update({"acousticness": beat["acousticness"]})
		beat_dict.update({"title": beat["title"]})
		beat_dict.update({"artist": beat["artist"]})
		beat_dict.update({"year": beat["year"]})
		beat_dict.update({"uri": beat["uri"]})
		beat_dict.update({"barks": bark_bands_over_thresh})
		beat_dict.update({"start": beat["others"][0]})
		beat_dict.update({"end": beat["others"][1]})
		beat_dict.update({"bpm": beat["others"][2]})
		beat_dict.update({"loudness": beat["others"][3]})
		beat_dict.update({"dissonance": beat["others"][4]})
		beat_dict.update({"onsetrate": beat["others"][5]})
		beat_dict.update({"mode_first": beat["chroma.first"][12]})
		beat_dict.update({"mode_fourth": beat["chroma.fourth"][12]})
		beat_dict.update({"coords": point_coords})
		all_beats.append(beat_dict)
		year = beat["year"]
		if year[-2] == '6' or year[-2] == '7':
			_70s_list.append(index)
		elif year[-2] == '8':
			_80s_list.append(index)
		elif year[-2] == '9':
			_90s_list.append(index)
		elif year[-2] == '0':
			_00s_list.append(index)
		elif year[-2] == '1':
			_10s_list.append(index)

	output_dict.update({"all_beats": all_beats})
	output_dict.update({"70s": _70s_list})
	output_dict.update({"80s": _80s_list})
	output_dict.update({"90s": _90s_list})
	output_dict.update({"00s": _00s_list})
	output_dict.update({"10s": _10s_list})

	with open(output_file_path, 'w') as outfile:
		json.dump(output_dict, outfile)


def get_euclidean_neighbors(point, beats, k_dimensions_map, number_of_neighbours, src_index):
	distances_to_point = []
	for i, tgt in enumerate(k_dimensions_map[:src_index] + k_dimensions_map[src_index+1:]):
		tgt = asarray(tgt)
		dist = norm(point - tgt)
		tgt_index = i
		if tgt_index >= src_index:
			tgt_index += 1
		src_uri = beats[src_index]["uri"]
		tgt_uri = beats[tgt_index]["uri"]

		# compute euclidean distance between point and tgt
		if (src_uri != tgt_uri):
			distances_to_point.append({"id": tgt_index, "distance": dist})
	distances_to_point.sort(key=itemgetter('distance'))
	n_neighbors = []
	for i in range(number_of_neighbours):
		n_neighbors.append(distances_to_point[i]["id"]) 
	return n_neighbors


def find_bands_over_thresh(beat_bark, bark_thresholds_vector):
	bands_over_thresh = 0
	for i in range(len(beat_bark)):
		if beat_bark[i] >= bark_thresholds_vector[i]:
			bands_over_thresh += 1
	return bands_over_thresh


def refine(src_index, beats, n_neighbors):
	# print n_neighbors
	refined_list = []
	src = beats[src_index]
	for tgt_index in n_neighbors:
		tgt = beats[tgt_index]
		dist = get_distance(src, tgt)
		refined_list.append({"id": tgt_index, "distance": dist})
	refined_list.sort(key=itemgetter('distance')) 
	refined_list_indexes = []
	for item in refined_list:
		refined_list_indexes.append(item["id"])
	# print refined_list_indexes
	return refined_list_indexes


def find_all_corresponding_pivots(beats, pivots_indexes):
	all_pivots = []
	for pivot_index in pivots_indexes:
		median_distant = get_median_for_pivot(beats, pivot_index)  #  median_distant_index should be the index of median_distant in beats
		median_distant_index = median_distant["index"]
		all_pivots.append({'first': pivot_index, 'second': median_distant_index, 'distance': median_distant["distance"]})
	return all_pivots


def get_median_for_pivot(beats, pivot_index):
	pivot = beats[pivot_index]
	distances_dict = []  
	# compute distance between pivot and all the others segments
	for index, beat in enumerate(beats[:pivot_index] + beats[pivot_index + 1:]):  # avoid calculating distance between pivot and itself
		distance = get_distance(pivot, beat)
		index_to_store = index
		if index_to_store >= pivot_index:
			index_to_store += 1
		distances_dict.append({"index":index_to_store, "distance": distance})
		global pivots_percentage
		pivots_percentage += (1./(len(beats) - 1))/dimensions * 100
		status(pivots_percentage)
	distances_dict.sort(key=itemgetter('distance')) 
	return distances_dict[len(distances_dict)/2]


def get_distance(src, tgt):
	src_mean = src["mfcc.mean"]
	src_covar = src["mfcc.covar"]
	src_invcovar = src["mfcc.invcovar"]
	dim = len(src_mean)
	src_mean = asarray(src_mean)
	src_covar = build_matrix_from_vector(src_covar, dim)
	src_invcovar = build_matrix_from_vector(src_invcovar, dim)
	tgt_mean = tgt["mfcc.mean"]
	tgt_covar = tgt["mfcc.covar"]
	tgt_invcovar = tgt["mfcc.invcovar"]
	tgt_mean = asarray(tgt_mean)
	tgt_covar = build_matrix_from_vector(tgt_covar, dim)
	tgt_invcovar = build_matrix_from_vector(tgt_invcovar, dim)
	skl_dist = SKL_distance(src_mean, tgt_mean, src_covar, tgt_covar, src_invcovar, tgt_invcovar, dim)
	sqrt_skl = sqrt(skl_dist)
	return sqrt_skl


def collect_all_beats_descriptors(files):
	beats = []
	all_loudnesses = []
	all_dissonances = []
	all_onsetsrates = []
	all_barks = []
	all_acousticnesses = []

	picked_indexes = []
	count = 0
	for file_ in files:
		with open(file_) as data_file:
			json_content = json.load(data_file)
		acousticness, title, artist, year = get_song_descriptors(json_content)
		all_acousticnesses.append(acousticness)

		for beat in json_content["beats.descriptors"]:
			beat.update({"acousticness": acousticness})
			beat.update({"title": title})
			beat.update({"artist": artist})
			beat.update({"year": year})
			beat.update({"uri": ('.').join(os.path.abspath(file_).split('.')[:-1]) + ".mp3"})
			beat.update({"id": count})
			try:
				beat["others"][3]
				all_loudnesses.append(beat["others"][3])
				all_dissonances.append(beat["others"][4])
				all_onsetsrates.append(beat["others"][5])
				all_barks.append(beat["barks"])
				beats.append(beat)
			except:
				continue
			count += 1
	all_loudnesses.sort()
	all_dissonances.sort()
	all_onsetsrates.sort()
	all_acousticnesses.sort()
	return beats, all_loudnesses, all_dissonances, all_onsetsrates, all_barks, all_acousticnesses


def get_random_pivots(beats):
	pivots_indexes = []
	len_beats = len(beats)
	while True:
		tmp_pivot = random.randrange(len_beats)
		if tmp_pivot not in pivots_indexes:
			pivots_indexes.append(tmp_pivot)
		if len(pivots_indexes) == dimensions:
			break
	return pivots_indexes

def status(percent):
	sys.stdout.write("%3d%%\r" % percent)
	sys.stdout.flush()


def get_song_descriptors(json_content):
	if json_content["song.acousticness"]: 
		acousticness = json_content["song.acousticness"]
	else: # acousticness could be 'None'
		acousticness = 0
	if json_content["song.title"]: 
		title = json_content["song.title"]
	else: 
		title = "N/A"
	if json_content["song.artist"]: 
		artist = json_content["song.artist"]
	else:
		artist = "N/A"
	if json_content["song.year"]: 
		year = json_content["song.year"]
	else: 
		year = "N/A"
	return acousticness, title, artist, year


def get_nearest_neighbors(src_index, beats, number_of_neighbours):
	src = beats[src_index]
	neighbors_indexes = []
	distances_to_point = []
	src_coords = src["coords"]	
	src_coords = asarray(src_coords)
	for tgt_index, tgt in enumerate((beats[:src_index] + beats[src_index + 1:])):
		tgt_coords = tgt["coords"]
		tgt_index_to_store = tgt_index
		if tgt_index >= src_index:
			tgt_index_to_store += 1
		src_uri = beats[src_index]["uri"]
		tgt_uri = beats[tgt_index_to_store]["uri"]
		if src_uri == tgt_uri:
			# we don't want to consider segments belonging to the same song
			continue
		# implement this to make the research faster
		# if (filters not satisfied):
		# 	continue
		tgt_coords = asarray(tgt_coords)
		dist = norm(src_coords - tgt_coords) # fastest method
		# compute euclidean distance between point and tgt
		distances_to_point.append({"id": tgt_index_to_store, "distance": dist})
	distances_to_point.sort(key=itemgetter('distance'))
	n_neighbors = []
	for i in range(number_of_neighbours):
		n_neighbors.append(distances_to_point[i]["id"]) 
	return n_neighbors


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Usage:", sys.argv[0], "<path_to_music_collection>"
		sys.exit(0)
	music_dir = sys.argv[1]
	if music_dir[-1] != "/":
		music_dir[-1] += "/"
	build_map(music_dir)
	# with open(music_dir + "/outputmap") as data_file:
	# 	json_content = json.load(data_file)
	# beats = json_content["all_beats"]
	# index = 2387
	# # print beats[index]
	# # print len(beats)
	# number_of_neighbours = int(len(beats) * filter_size)
	# start_time = time.time()
	# nnn_1 = get_nearest_neighbors(index, beats, number_of_neighbours)
	# time_1 = time.time() - start_time
	# print "Seconds for computing neighbors for one point:", time_1