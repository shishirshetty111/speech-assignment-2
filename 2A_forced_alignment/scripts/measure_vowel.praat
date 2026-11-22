# Print F1 and F2 (Hz) at the midpoint of an interval (Burg formant analysis), or with
# start = end = 0 print the median F0 (Hz) of the whole file, used to choose the formant ceiling.
# Praat.exe --run measure_vowel.praat <wav> <start> <end> <max_formant_hz>
form Measure vowel
    sentence wav
    real start
    real end
    real max_formant 5500
endform
sound = Read from file: wav$
if start = 0 and end = 0
    pitch = To Pitch: 0, 75, 500
    f0 = Get quantile: 0, 0, 0.5, "Hertz"
    writeInfoLine: fixed$(f0, 0)
else
    formant = To Formant (burg): 0, 5, max_formant, 0.025, 50
    mid = (start + end) / 2
    f1 = Get value at time: 1, mid, "hertz", "linear"
    f2 = Get value at time: 2, mid, "hertz", "linear"
    writeInfoLine: fixed$(f1, 0), " ", fixed$(f2, 0)
endif
