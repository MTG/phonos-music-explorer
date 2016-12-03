# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import os.path
from os import listdir
import random
import unicodedata
from numpy import sqrt, asarray
from numpy.linalg import norm
import subprocess
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
phonos = True

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
# enable streaming through http
streaming = True

# Determines when current player will stop playing current song
deadline = 0

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
currently_playing = False
loading_finished = False
useAlsaSink = False

# Gstreamer initialization
GObject.threads_init()
Gst.init(None)

# Gst.debug_set_active(True)
# os.environ["GST_DEBUG"] = "*:4"
# os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "/tmp"
# os.environ["GST_DEBUG_FILE"] = "/tmp/gst.log"

# starting_point = time.time()


def fix_song_path(song_path):
    return song_path.replace(u'/home/giuseppe/empd/app/phonos/', u'/phonos/')


class CustomBin(Gst.Bin):
    def __init__(self, audiofile):
        Gst.Bin.__init__(self)
        self.uridecodebin = Gst.ElementFactory.make("uridecodebin")
        self.uridecodebin.set_property("uri", audiofile)
        # uridecodebin_caps = Gst.Caps.from_string("audio/x-raw")
        uridecodebin_caps = Gst.Caps.from_string("{audio}, format={format}, layout={layout}, rate={rate}".format(
            audio='audio/x-raw', format='(string)S32LE',  layout='(string)interleaved',
            rate='(int){ 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000 }'
        ))
        self.uridecodebin.set_property("caps", uridecodebin_caps)
        self.volume = Gst.ElementFactory.make("volume")
        self.volume.set_property("volume", 0)
        self.audioconvert = Gst.ElementFactory.make("audioconvert")
        self.audioresample = Gst.ElementFactory.make("audioresample")
        self.audioresample.set_property("quality", 10)
        self.add(self.audioconvert)
        self.add(self.audioresample)
        self.add(self.uridecodebin)
        self.add(self.volume)
        self.volume.link(self.audioconvert)
        self.audioconvert.link(self.audioresample)
        self.uridecodebin.connect("pad-added", self.pad_added_handler)
        self.audioresample_pad = self.audioresample.get_static_pad('src')
        self.ghostpad = Gst.GhostPad.new("src", self.audioresample_pad)
        self.add_pad(self.ghostpad)

    def pad_added_handler(self, src, new_pad):
        # print("Receiving pad '%s' from '%s':" % (new_pad.get_name(), src.get_name()))
        if new_pad.is_linked():
            print("Pad already linked")
            return
        new_pad_type = new_pad.query_caps(None).to_string()
        if not new_pad_type.startswith("audio/x-raw"):
            print("It has type '%s' which is not raw audio. Ignoring")
            return
        ret = new_pad.link(self.volume.get_static_pad("sink"))
        # print ret
        return

    def change_volume(self, new_vol):
        self.volume.set_property("volume", new_vol)

    def seek_bin(self, inpoint, endpoint):
        self.set_state(Gst.State.PAUSED)
        self.get_state(Gst.CLOCK_TIME_NONE)
        self.seek(1, Gst.Format.TIME, Gst.SeekFlags.KEY_UNIT, Gst.SeekType.SET, inpoint * Gst.SECOND,
                  Gst.SeekType.NONE, 0)
        self.set_state(Gst.State.PLAYING)


class CustomPipeline(Gst.Pipeline):
    def __init__(self):
        """
        Pipeline with two custom bins and other elements necessary for the playback
        """
        Gst.Pipeline.__init__(self)
        bus = self.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)
        self.adder = Gst.ElementFactory.make("adder")
        self.volume = Gst.ElementFactory.make("volume")
        self.volume.set_property("volume", 1)
        self.add(self.adder)
        self.add(self.volume)
        self.adder.link(self.volume)
        if useAlsaSink:
            self.alsasink = Gst.ElementFactory.make("alsasink")
            self.add(self.alsasink)
            self.volume.link(self.alsasink)
        else:
            self.lame = Gst.ElementFactory.make("lamemp3enc")
            self.add(self.lame)
            self.volume.link(self.lame)
            self.tcpsink = Gst.ElementFactory.make("udpsink")
            self.tcpsink.set_property("port", 8070)
            self.add(self.tcpsink)
            self.lame.link(self.tcpsink)

    def add_bin(self, uri_audiofile):
        tmp_bin = CustomBin(uri_audiofile)
        self.add(tmp_bin)
        src_tmp_bin = tmp_bin.get_static_pad("src")
        sink_adder = self.adder.get_request_pad("sink_%u")
        src_tmp_bin.link(sink_adder)
        return tmp_bin, src_tmp_bin, sink_adder

    def on_message(self, bus, message):
        global currently_playing
        t = message.type
        if t == Gst.MessageType.EOS:
            self.set_state(Gst.State.NULL)
            currently_playing = False
        elif t == Gst.MessageType.ERROR:
            self.set_state(Gst.State.NULL)
            err, debug = message.parse_error()
            print("Custom error: %s" % err, debug)
            currently_playing = False
        else:
            print(t)

    def start(self):
        bin2 = None
        global currently_playing
        currently_playing = True
        self.set_state(Gst.State.PLAYING)
        while True:
            if currently_playing:
                (uri_from_queue, title_from_queue, artist_from_queue, year_from_queue,
                 inpoint, endpoint) = self.pick_next_song()
                self.update_global_values(uri_from_queue, title_from_queue, artist_from_queue, year_from_queue)
                bin1, bin1_src, sink1_adder = self.add_bin(uri_from_queue)
                playback_inpoint = max(inpoint - CROSSFADE, 0)
                segment_duration = endpoint - playback_inpoint
                global deadline
                deadline = time.time() + segment_duration
                bin1.seek_bin(max(inpoint - CROSSFADE, 0), endpoint)
                self.crossfade(bin1, bin2)
                # self.print_pipeline()
                time_to_play = deadline - time.time() - CROSSFADE
                # print time.time() - starting_point, " Playing", uri_from_queue
                if time_to_play > 0:
                    time.sleep(time_to_play)
                if bin2:
                    self.remove_bin(bin2, bin2_src, sink2_adder)
                if not currently_playing:
                    continue
                (uri_from_queue, title_from_queue, artist_from_queue, year_from_queue,
                 inpoint, endpoint) = self.pick_next_song()
                self.update_global_values(uri_from_queue, title_from_queue, artist_from_queue, year_from_queue)
                bin2, bin2_src, sink2_adder = self.add_bin(uri_from_queue)
                playback_inpoint = max(inpoint - CROSSFADE, 0)
                segment_duration = endpoint - playback_inpoint
                global deadline
                deadline = time.time() + segment_duration
                bin2.seek_bin(max(inpoint - CROSSFADE, 0), endpoint)
                self.crossfade(bin2, bin1)
                time_to_play = deadline - time.time() - CROSSFADE
                # print time.time() - starting_point, " Playing", uri_from_queue
                if time_to_play > 0:
                    time.sleep(time_to_play)
                self.remove_bin(bin1, bin1_src, sink1_adder)
            else:
                time.sleep(0.2)

    def remove_bin(self, bin_, bin_src, adder_sink):
        bin_src.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, self.event_probe, adder_sink)
        bin_src.unlink(adder_sink)
        bin_.set_state(Gst.State.NULL)
        self.adder.release_request_pad(adder_sink)
        self.remove(bin_)

    def event_probe(self, pad, info, ud):
        pass

    @staticmethod
    def crossfade(fader_in, fader_out):
        fader_in_volume = 0
        fader_out_volume = 1
        for i in range(10):
            if fader_in and fader_out:
                time.sleep(CROSSFADE / 10.)
                fader_in_volume += 0.1
                fader_out_volume -= 0.1
                fader_in.change_volume(fader_in_volume)
                fader_out.change_volume(fader_out_volume)
            elif fader_in:
                time.sleep(CROSSFADE / 10.)
                fader_in_volume += 0.1
                fader_in.change_volume(fader_in_volume)
            else:
                time.sleep(CROSSFADE / 10.)
                fader_out_volume -= 0.1
                fader_out.change_volume(fader_out_volume)

    def change_volume(self, volume):
        self.volume.set_property("volume", volume)

    def pause(self):
        self.set_state(Gst.State.PAUSED)
        global currently_playing
        currently_playing = False

    def resume(self):
        self.set_state(Gst.State.PLAYING)
        global currently_playing
        currently_playing = True

    def print_pipeline(self):
        dotfile = "/tmp/pipeline.dot"
        pngfile = "/tmp/pipeline.png"
        if os.access(dotfile, os.F_OK):
            os.remove(dotfile)
        if os.access(pngfile, os.F_OK):
            os.remove(pngfile)
        Gst.debug_bin_to_dot_file(self, Gst.DebugGraphDetails.ALL, "pipeline")
        os.system("dot -Tpng '%s' > '%s'" % (dotfile, pngfile))

    @staticmethod
    def pick_next_song():
        """
        Pick the next song (if present) from the queue. Updates player's properties.
        """

        while True:
            try:
                uri_from_queue = playlist[songs_played][0]
                title_from_queue = playlist[songs_played][1]
                artist_from_queue = playlist[songs_played][2]
                year_from_queue = playlist[songs_played][3]
                inpoint = playlist[songs_played][4]
                endpoint = playlist[songs_played][5]
                break
            except:
                time.sleep(.1)
                continue
        return uri_from_queue, title_from_queue, artist_from_queue, year_from_queue, inpoint, endpoint

    @staticmethod
    def update_global_values(uri, title, artist, year):
        """
        Sets everything ready for the playback.
        """

        global currently_playing_song, current_artist, current_track, current_year, songs_played
        currently_playing_song = uri
        current_track = title
        current_artist = artist
        current_year = year
        songs_played += 1


class Play:
    def __init__(self):
        self.custom_player = CustomPipeline()
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

        self.ignoreSimilarities = False
        self.forbidden_songs = {}
        self.bars = 4
        self.forceAccept = False
        self.use_loudness = False
        self.use_noisiness = False
        self.use_rhythm = False
        self.use_bands = False
        self.use_acousticness = False
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

        print(self.slider1_value, self.slider2_value, self.slider3_value, self.slider4_value, self.slider5_value,
              self.slider6_value, self.slider7_value, self.slider8_value, self.slider9_value, self.slider10_value)
        self.slidersChanged = True

    def activate_sliders(self, use_loudness, use_noisiness, use_rhythm, use_bands, use_acousticness):
        """
        Function called when the user interacts with button for activating sliders.
        """

        if use_loudness == 'true':
            self.use_loudness = True
        else:
            self.use_loudness = False

        if use_noisiness == 'true':
            self.use_noisiness = True
        else:
            self.use_noisiness = False

        if use_rhythm == 'true':
            self.use_rhythm = True
        else:
            self.use_rhythm = False

        if use_bands == 'true':
            self.use_bands = True
        else:
            self.use_bands = False

        if use_acousticness == 'true':
            self.use_acousticness = True
        else:
            self.use_acousticness = False

        self.slidersChanged = True

    def change_segment_length(self, length):
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
        self.empty_unplayed_songs_queue()
        self.ignoreSimilarities = True

    def play_request(self, playing):
        """
        Handles user interaction with playback buttons.
        """
        global currently_playing
        if playing == "false":
            self.custom_player.pause()
        else:
            global last_start
            last_start = time.time()
            if songs_played == 0:
                self.start_player()
                if streaming:
                    # redirect streaming coming from tcpsink to port 8080
                    command = 'vlc tcp://127.0.0.1:8070 -d :sout=#http{dst=:8080/} :sout-keep vlc://quit'
                    command2 = 'vlc tcp://127.0.0.1:8080 -d vlc://quit'
                else:
                    # play streaming coming from tcpsink
                    command = 'vlc tcp://127.0.0.1:8070 -d vlc://quit'
                subprocess.call(command, shell=True)
                subprocess.call(command2, shell=True)
            else:
                self.custom_player.resume()
            if self.on_automatic_pause:
                self.on_automatic_pause = False

    def update_volume(self, volume):
        new_vol = int(volume) / 100.
        self.custom_player.change_volume(new_vol)

    @staticmethod
    def event_stream():
        """
        Communicates a change in the playback to the user (a new song is playing).
        """
        global current_track, current_artist, current_year

        previous_playing_song = None
        while True:
            if currently_playing_song != previous_playing_song:
                # dirty trick to make the user less conscious of the delay in audio streaming
                if streaming:
                    time.sleep(5)
                else:
                    time.sleep(1.5)
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
        Communicates an automatic pause to the playback, so that the GUI can be updated
        """
        previous_state = False
        while True:
            if self.on_automatic_pause != previous_state:
                yield 'data: %s\n\n' % self.on_automatic_pause
                previous_state = self.on_automatic_pause
            time.sleep(.2)

    @staticmethod
    def loading():
        """
        Communicates an automatic pause to the playback, so that the GUI can be updated
        """
        global loading_finished
        while True:
            if loading_finished:
                yield 'data: %s\n\n' % loading_finished
                loading_finished = False  # to prevent activating this script again
                break
            else:
                time.sleep(0.2)

    def main(self):
        """
        Loads the map for selected directory of music and calls the method for creating a playlist for it.
        """
        if phonos:
            self.music_dir = "/phonos/"
        else:
            self.music_dir = "./music/"

        if len(sys.argv) > 1:
            self.music_dir = os.path.abspath(sys.argv[1])
        if self.music_dir[-1] != "/":
            self.music_dir += "/"

        try:
            (self.beats_map, self._70s_songs, self._80s_songs, self._90s_songs,
             self._00s_songs, self._10s_songs) = self.load_map()
            self.fix_map_uri_paths()
            self.setup_sliders_values()
        except:
            print("There were issues loading the map of musical segments. Please build one for your collection.",
                  sys.stderr)
            return
        self.create_playlist()

    def fix_map_uri_paths(self):
        for segment in self.beats_map:
            segment['uri'] = fix_song_path(segment['uri'])

    def create_playlist(self):
        """
        Randomly picks first song and creates a playlist for it.
        """
        self.initialize_playlist_creation()
        is_first_song = True
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
        for i in range(self.bars):
            tmp_index = last_segment_picked_index + i
            if last_song_picked == self.beats_map[tmp_index]["uri"]:
                next_segment_index_end = tmp_index
            else:
                break

        end_time = self.beats_map[next_segment_index_end]["end"]
        print("Added: {song_added} [{counter}], interval: [{start}, {end}], duration: {duration}".format(
            song_added=last_song_picked.encode('utf-8'), counter=self.counter, start=start_time, end=end_time,
            duration=end_time - start_time
        ), file=sys.stderr)

        # put the selected song (or segment) into "forbidden_songs", just to try to avoid to play it again very soon
        if force_different_consecutive_songs:
            self.update_forbidden_songs_queue(last_song_picked)
        else:
            self.update_forbidden_songs_queue(last_segment_picked_index)

        # put the selected song into the playback queue
        playlist[self.counter] = ["file://" + last_song_picked, song_title, song_artist, song_year, start_time,
                                  end_time]
        global loading_finished
        loading_finished = True

        # find all the other segments to be put in the playlist
        while True:
            self.clean_playlist_queue()
            self.perform_automatic_pause_if_necessary()
            self.check_for_changes()
            if self.is_deadline_incoming():
                self.ignoreSimilarities = True

            # let the cpu rest (avoid overheating) if we still have to play more than 6 elements of the playlist
            if self.counter - songs_played >= 5:
                time.sleep(1)
                continue

            next_segment_index = self.pick_next_segment(last_segment_picked_index)
            next_segment = self.beats_map[next_segment_index]

            last_segment_picked_index = next_segment_index
            last_song_picked = next_segment["uri"]
            for i in range(self.bars):
                tmp_index = next_segment_index + i
                try:
                    if last_song_picked == self.beats_map[tmp_index]["uri"]:
                        next_segment_index_end = tmp_index
                    else:
                        break
                except:
                    next_segment_index_end = next_segment_index
                    break

            # compare next segments with the last bar of the current one
            last_segment_picked_index = next_segment_index_end

            if force_different_consecutive_songs:
                self.update_forbidden_songs_queue(last_song_picked)
            else:
                self.update_forbidden_songs_queue(last_segment_picked_index)
            start_time = next_segment["start"]
            end_time = self.beats_map[next_segment_index_end]["end"]
            title = next_segment["title"]
            artist = next_segment["artist"]
            year = next_segment["year"]
            self.ignoreSimilarities = False
            self.counter += 1
            # print "\033[01;36m", time.time() - starting_point, " Song added:", last_song_picked, "\033[0m"
            print("Added: {song_added} [{counter}], interval: [{start}, {end}], duration: {duration}".format(
                song_added=last_song_picked.encode('utf-8'), counter=self.counter, start=start_time, end=end_time,
                duration=end_time - start_time
            ), file=sys.stderr)
            # if is_first_song:
            # 	self.start_player()
            # 	is_first_song = False
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
        filtered_suitable_segments = self.filter_list(suitable_segments, current_bpm, current_loudness,
                                                      current_mode, bpm_thresh, loudness_thresh)
        # filtered candidates list could be empty
        if not filtered_suitable_segments:
            filtered_suitable_segments = suitable_segments

        # print "Time to perform filtering:", time.time() - start_1
        current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar = self.load_mfccs(current_segment,
                                                                                       current_song_beats)
        if len(filtered_suitable_segments) > 60:
            number_of_neighbors = self.select_number_of_neighbors(filtered_suitable_segments)
            neighbors = self.find_nearest_neighbors(filtered_suitable_segments, current_segment,
                                                    number_of_neighbors)
            neighbors.sort()
            good_segments, analyzed_segments = self.get_suitable_segments_in_neighbors(neighbors, current_mfcc_mean,
                                                                                       current_mfcc_covar,
                                                                                       current_mfcc_invcovar,
                                                                                       current_segment)
        else:
            good_segments, analyzed_segments = self.get_suitable_segments_in_neighbors(filtered_suitable_segments,
                                                                                       current_mfcc_mean,
                                                                                       current_mfcc_covar,
                                                                                       current_mfcc_invcovar,
                                                                                       current_segment)
        if good_segments:
            random_index = random.randrange(len(good_segments))
            return good_segments[random_index]
        else:
            analyzed_segments.sort(key=itemgetter('distance'))
            random_index = random.randrange(math.ceil(len(analyzed_segments) / 10.))
            return analyzed_segments[random_index]["index"]

    def get_suitable_segments_in_neighbors(self, neighbors, current_mfcc_mean, current_mfcc_covar,
                                           current_mfcc_invcovar, current_segment):

        analyzed_segments = []
        # list of all segments whose timbre is really similar to the current one
        good_segments = []

        target_song_beats = []
        previous_uri = ""
        for neighbor_index in neighbors:
            # avoid computing distance again if we have already computed that for this segment
            distance_previously_computed = self.neighbor_already_analyzed(neighbor_index, analyzed_segments)
            if distance_previously_computed:  # equivalent to say "if this neighbor has already been analyzed"
                analyzed_segments.append({"index": neighbor_index, "distance": distance_previously_computed})
                continue

            neighbor = self.beats_map[neighbor_index]

            if self.ignoreSimilarities:
                euclidean_distance = norm(asarray(current_segment["coords"]) - asarray(neighbor["coords"]))
                analyzed_segments.append({"index": neighbor_index, "distance": euclidean_distance})
                continue

            # now we need to load the descriptor file for the song to get the mfcc values
            if neighbor["uri"] != previous_uri:
                target_song_beats = self.load_json_descriptors(neighbor)
                previous_uri = neighbor["uri"]

            neighbor_mfcc_mean, neighbor_mfcc_covar, neighbor_mfcc_invcovar = self.load_mfccs(neighbor,
                                                                                              target_song_beats)
            skl_thresh = 20
            skl_dist = self.compute_skl_dist(current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar,
                                             neighbor_mfcc_mean, neighbor_mfcc_covar, neighbor_mfcc_invcovar)
            if skl_dist <= skl_thresh:
                good_segments.append(neighbor_index)
            else:
                analyzed_segments.append({"index": neighbor_index, "distance": skl_dist})

        return good_segments, analyzed_segments

    @staticmethod
    def random_subsampling(candidates_list):
        input_size = len(candidates_list)
        output_size = int(500 * (math.log(input_size, 500)) ** 2.5)
        output_list = []
        for i in range(output_size):
            random_index = random.randrange(len(candidates_list))
            candidate = candidates_list[random_index]
            output_list.append(candidate)
        return output_list

    @staticmethod
    def load_mfccs(segment, song_beats):
        for beat in song_beats:
            if abs(segment["start"] - beat["others"][0]) <= 0.0001:
                segment_mfcc_mean = beat["mfcc.mean"]
                segment_mfcc_covar = beat["mfcc.covar"]
                segment_mfcc_invcovar = beat["mfcc.invcovar"]
                return segment_mfcc_mean, segment_mfcc_covar, segment_mfcc_invcovar

    @staticmethod
    def get_comparable_features(segment):
        segment_bpm = segment["bpm"]
        segment_loudness = segment["loudness"]
        segment_mode = segment["mode_fourth"]
        return segment_bpm, segment_loudness, segment_mode

    @staticmethod
    def get_thresholds():
        bpm_thresh = 3
        loudness_thresh = 5
        return bpm_thresh, loudness_thresh

    def filter_list(self, candidates_list, current_bpm, current_loudness, current_mode, bpm_thresh, loudness_thresh):
        output_list = candidates_list[:]
        for candidate_index in candidates_list:
            candidate_segment = self.beats_map[candidate_index]
            candidate_bpm, candidate_loudness, candidate_mode = self.get_comparable_features(candidate_segment)
            if not key_distance(current_mode, candidate_mode) or (
                bpm_distance(current_bpm, candidate_bpm) > bpm_thresh) or \
                    (abs(current_loudness - candidate_loudness) > loudness_thresh):
                output_list.remove(candidate_index)
        return output_list

    def select_number_of_neighbors(self, candidates_list):
        if len(candidates_list) > len(self.beats_map):
            number_of_neighbors = int(filter_size * len(self.beats_map))
        else:
            number_of_neighbors = int(filter_size * len(candidates_list))
        return number_of_neighbors

    @staticmethod
    def compute_skl_dist(current_mfcc_mean, current_mfcc_covar, current_mfcc_invcovar, neighbor_mfcc_mean,
                         neighbor_mfcc_covar, neighbor_mfcc_invcovar):
        dim = len(current_mfcc_mean)
        src_mean = asarray(current_mfcc_mean)
        src_covar = build_matrix_from_vector(current_mfcc_covar, dim)
        src_invcovar = build_matrix_from_vector(current_mfcc_invcovar, dim)
        tgt_mean = asarray(neighbor_mfcc_mean)
        tgt_covar = build_matrix_from_vector(neighbor_mfcc_covar, dim)
        tgt_invcovar = build_matrix_from_vector(neighbor_mfcc_invcovar, dim)
        skl_dist = SKL_distance(src_mean, tgt_mean, src_covar, tgt_covar, src_invcovar, tgt_invcovar, dim)
        return skl_dist

    @staticmethod
    def load_json_descriptors(segment):
        song_path = fix_song_path(segment["uri"])
        song_json_path = ('.'.join(song_path.split('.')[:-1]) + ".json")
        song_json_path = unicodedata.normalize("NFKD", song_json_path)
        with open(song_json_path.encode("utf-8")) as data_file:
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

    @staticmethod
    def neighbor_already_analyzed(neighbor_index, analyzed_segments):
        for segment in analyzed_segments:
            if segment["index"] == neighbor_index:
                return segment["distance"]
        return None

    def find_segments_suitable_to_sliders(self):
        """
        Returns a list of segments that observe sliders' values and whose length is greater than 2 * crossfade value.
        """
        # print "Loudness threshs:", self.loudness_lower_thresh, self.loudness_upper_thresh
        # print "Noisiness threshs:", self.dissonance_lower_thresh, self.dissonance_upper_thresh
        # print "Onsets threshs:", self.onsetrate_lower_thresh, self.onsetrate_upper_thresh
        # print "Acousticness threshs:", self.acousticness_lower_thresh, self.acousticness_upper_thresh
        suitable_segments = []
        almost_suitable_segments = []
        random_segments = []
        pick_from_70s = self.slider1_value >= 1
        pick_from_80s = self.slider2_value >= 1
        pick_from_90s = self.slider3_value >= 1
        pick_from_00s = self.slider4_value >= 1
        pick_from_10s = self.slider5_value >= 1
        if pick_from_70s:
            for beat_index in self._70s_songs:
                beat = self.beats_map[beat_index]
                if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
                    self.add_if_suitable(beat, self.slider1_value, beat_index, suitable_segments,
                                         almost_suitable_segments)
                    random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
        if pick_from_80s:
            for beat_index in self._80s_songs:
                beat = self.beats_map[beat_index]
                if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
                    self.add_if_suitable(beat, self.slider2_value, beat_index, suitable_segments,
                                         almost_suitable_segments)
                    random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
        if pick_from_90s:
            for beat_index in self._90s_songs:
                beat = self.beats_map[beat_index]
                if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
                    self.add_if_suitable(beat, self.slider3_value, beat_index, suitable_segments,
                                         almost_suitable_segments)
                    random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
        if pick_from_00s:
            for beat_index in self._00s_songs:
                beat = self.beats_map[beat_index]
                if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
                    self.add_if_suitable(beat, self.slider4_value, beat_index, suitable_segments,
                                         almost_suitable_segments)
                    random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
        if pick_from_10s:
            for beat_index in self._10s_songs:
                beat = self.beats_map[beat_index]
                if (beat["end"] - beat["start"] > 2 * CROSSFADE) and self.not_a_repetition(beat, beat_index):
                    self.add_if_suitable(beat, self.slider5_value, beat_index, suitable_segments,
                                         almost_suitable_segments)
                    random_segments.append({"index": beat_index, "distance": self.distance_to_slider_conf(beat)})
        if suitable_segments:
            # print "Case #1"
            return suitable_segments
        elif almost_suitable_segments:
            # print "Case #2"
            return almost_suitable_segments
        else:
            # print "Case #3"
            random_segments.sort(key=itemgetter('distance'))
            segments_found = []
            for segment in random_segments[:int(math.ceil(len(random_segments) / 1000.))]:
                segments_found.append(segment["index"])
            return segments_found

    def add_if_suitable(self, beat, slider_value, beat_index, suitable_segments, almost_suitable_segments):
        (beat_loudness, beat_dissonance, beat_onsetrate, beat_barks,
         beat_acousticness, beat_uri) = self.get_relevant_features(beat)
        if self.are_sliders_strictly_satisfied(beat_loudness, beat_dissonance, beat_onsetrate, beat_barks,
                                               beat_acousticness):
            for i in range(slider_value):
                suitable_segments.append(beat_index)
        elif self.are_sliders_almost_satisfied(beat_loudness, beat_dissonance, beat_onsetrate, beat_barks,
                                               beat_acousticness):
            for i in range(slider_value):
                almost_suitable_segments.append(beat_index)

    def not_a_repetition(self, beat, beat_index):
        return not avoid_repetitions or (
            (force_different_consecutive_songs and beat["uri"] not in self.forbidden_songs.keys()) or
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
        dist = loudness_distance ** 2 + dissonance_distance ** 2 + onsetrate_distance ** 2 + acoustic_distance ** 2
        return sqrt(dist)

    @staticmethod
    def is_deadline_incoming():
        deadline_incoming = (len(playlist) <= 1) and (deadline - time.time() < 5)
        return deadline_incoming

    def load_map(self):
        with open(self.music_dir + "outputmap") as data_file:
            json_content = json.load(data_file)
        return json_content["all_beats"], json_content["70s"], json_content["80s"], json_content["90s"], json_content[
            "00s"], json_content["10s"]

    @staticmethod
    def get_relevant_features(beat):
        beat_loudness = beat["loudness"]
        beat_dissonance = beat["dissonance"]
        beat_onsetrate = beat["onsetrate"]
        beat_barks = beat["barks"]
        beat_acousticness = beat["acousticness"]
        beat_uri = beat["uri"]
        return beat_loudness, beat_dissonance, beat_onsetrate, beat_barks, beat_acousticness, beat_uri

    def setup_sliders_values(self):
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

    def adapt_thresholds_to_sliders(self):
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

    def are_sliders_strictly_satisfied(self, actual_loudness, actual_dissonance, actual_onsetrate,
                                       bark_bands_over_thresh, song_acousticness):
        loudness_satisfied = (not self.use_loudness or (
            self.loudness_lower_thresh <= actual_loudness <= self.loudness_upper_thresh))
        noisiness_satisfied = (not self.use_noisiness or (
            self.dissonance_lower_thresh <= actual_dissonance <= self.dissonance_upper_thresh))
        onsetrate_satisfied = (not self.use_rhythm or (
            self.onsetrate_lower_thresh <= actual_onsetrate <= self.onsetrate_upper_thresh))
        barks_satisfied = (not self.use_bands or (
            self.barks_lower_thresh <= bark_bands_over_thresh <= self.barks_upper_thresh))
        acousticness_satisfied = (not self.use_acousticness or (
            self.acousticness_lower_thresh <= song_acousticness <= self.acousticness_upper_thresh))

        strict_similarity = (loudness_satisfied and noisiness_satisfied and onsetrate_satisfied
                             and barks_satisfied and acousticness_satisfied)
        return strict_similarity

    def are_sliders_almost_satisfied(self, actual_loudness, actual_dissonance, actual_onsetrate, bark_bands_over_thresh,
                                     song_acousticness):
        loudness_almost_satisfied = (not self.use_loudness or (
            self.loudness_safe_lower_thresh <= actual_loudness <= self.loudness_safe_upper_thresh))
        noisiness_almost_satisfied = (not self.use_noisiness or (
            self.dissonance_safe_lower_thresh <= actual_dissonance <= self.dissonance_safe_upper_thresh))
        onsetrate_almost_satisfied = (not self.use_rhythm or (
            self.onsetrate_safe_lower_thresh <= actual_onsetrate <= self.onsetrate_safe_upper_thresh))
        barks_almost_satisfied = (not self.use_bands or (
            self.barks_safe_lower_thresh <= bark_bands_over_thresh <= self.barks_safe_upper_thresh))
        acousticness_almost_satisfied = (not self.use_acousticness or (
            self.acousticness_safe_lower_thresh <= song_acousticness <= self.acousticness_safe_upper_thresh))
        approx_similarity = (loudness_almost_satisfied and noisiness_almost_satisfied and onsetrate_almost_satisfied and
                             barks_almost_satisfied and acousticness_almost_satisfied)
        return approx_similarity

    def update_decades_sliders(self):
        """
        If all the decades sliders have the same value, just put them to 1 (so that weighted queue will be shorter)
        """
        if (self.slider1_value == self.slider2_value) and (self.slider2_value == self.slider3_value) and (
            self.slider3_value == self.slider4_value) and \
                (self.slider4_value == self.slider5_value):
            self.slider1_value = 1
            self.slider2_value = 1
            self.slider3_value = 1
            self.slider4_value = 1
            self.slider5_value = 1

    def check_for_changes(self):
        """
        Check if some sliders have been changed by the user.
        If so, make proper changes to values.
        """
        if self.slidersChanged:
            self.adapt_thresholds_to_sliders()
            self.update_decades_sliders()
            self.empty_unplayed_songs_queue()
            self.ignoreSimilarities = True
            self.slidersChanged = False

    def empty_unplayed_songs_queue(self):
        """
        Empties the playlist queue.
        """
        # empty the queue of songs that haven't been played yet
        if len(playlist) >= 1:
            for i in playlist.keys():
                if i >= songs_played - 1:
                    del playlist[i]
        self.counter = songs_played - 1

    @staticmethod
    def clean_playlist_queue():
        """
        Delete useless items of playlist queue in order to avoid memory leaking.
        """

        global songs_played
        for i in playlist.keys():
            if i < songs_played:
                del playlist[i]

    def update_forbidden_songs_queue(self, candidate_song):
        """
        Updates values in forbidden songs queue (that is the list of songs that cannot be selected because they have
        been used recently).
        Reduces by 1 the number of turns on which they will still be forbidden.
        """
        for key in self.forbidden_songs.keys():
            val = self.forbidden_songs[key]
            val -= 1
            if val == 0:
                self.forbidden_songs.pop(key, None)
            else:
                self.forbidden_songs.update({key: val})
        self.forbidden_songs.update({candidate_song: forbidden_value})

    def perform_automatic_pause_if_necessary(self):
        """
        Automatically performs a stop of 'automatic_pause_length' seconds each 'automatic_pause_length' seconds.
        """
        global last_start, currently_playing
        if currently_playing and automatic_pause_length != 0 and (time.time() - last_start > automatic_pause_length):
            currently_playing = False
            self.on_automatic_pause = True
            self.player.pause()
            self.player2.pause()
            need_to_restart = True
            for i in range(automatic_pause_length * 10):
                time.sleep(0.1)
                if currently_playing:  # check if the user have resumed the playback manually
                    need_to_restart = False
                    break
            if need_to_restart:
                global last_start
                last_start = time.time()
                currently_playing = True
                self.on_automatic_pause = False
                self.player.resume()
                self.player2.resume()

    def initialize_playlist_creation(self):
        """
        Initialize variables so that create_playlist() can run properly.
        """
        global songs_played
        songs_played = 0
        self.counter = 0

    def start_player(self):
        """
        Starts the two audio players, that will be playing together in order to achieve a crossfade effect.
        """
        thr = threading.Thread(target=self.custom_player.start, args=())
        thr.start()
