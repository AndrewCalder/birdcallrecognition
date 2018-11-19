#C S 4033-001 Machine Learning
#Short SL Project - Bird Call Recognition
#David McKnight, Cody Degner, Andrew Calder

#Import packages
import librosa;

#Initialize variables
#Training and validation data in these folders
trainingDataPaths = ["data_cardinal", "data_crow"];
#Testing data (mixed bird species) in this folder?
testingDataPath = "data_testing";

#Sample sound file
test = "test.wav";
test2 = trainingDataPaths[0] + "/XC171534-cardinalsong.mp3";


# Beat tracking example
#from __future__ import print_function
#import librosa
# 1. Get the file path to the included audio example
filename = test
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename)
# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)



#Load all data
print("Loading data...");
#TODO

#Split training data into training and validation sets
print("Splitting data...");
#TODO
#Subdivide audio files into 10-second-or-so samples?

#Perform training
print("SL training...");
#TODO

#Perform validation
print("SL validation...");
#TODO

#Perform testing
print("Testing...");
#TODO
