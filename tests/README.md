# Testing
This directory contains Jupyter notebooks with tests.

## Onset detection
There are no specific tests for onset detection. Onset detection approaches are compared by comparing the results 
of the following steps using different onset detection approaches. 

## Tempo analysis
Tempo analysis testing can be found in file `test-tempo.ipynb`. Results of these tests are in the file `tempograms.csv`. 
This file contains the correct tempo and then calculated tempi from each combination of approaches.

## Beat detection
Beat detection testing file is `test-beat.ipynb`. This file generates audio files from provided songs, and it adds 
the calculated beat track over each song. 

Testing was done by listening to these generated files and classifying the results in the following way:
* **ok** - if the generated beat track aligned with beats
* **half** - if the generated beat clicks were right in the middle of two actual beats, so the found time 
 shift was exactly half of the correct beat time shift
* *no** - if the beat track was completely wrong

The results can be found in `beat_results.csv`. The score-based approach was slightly better. As for onset detection 
approach, energy-based novelty function gave slightly better results.

## Rhythm detection
Rhythm detection testing file is `test-rhythm.ipynb`. This file generates audio files from provided songs, and it adds 
the created rhythm track over each song. 
Testing was done by listening to these generated files. But I did not classify the results, as it can be very 
subjective. 

Overall, there were good results for songs with simple rhythmic patterns. The equal song partitioning worked better 
than chorus-verse partitioning.

