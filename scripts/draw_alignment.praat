# Draw waveform, spectrogram and all tiers of a TextGrid (tier names on the left), save as PNG.
# Run headless:  Praat.exe --run draw_alignment.praat <wav> <TextGrid> <png> <tmin> <tmax> <title>
# (tmax = 0 draws the whole file). The picture grows with the number of tiers.

form Draw alignment
    sentence wav
    sentence tg
    sentence png
    real tmin 0
    real tmax 0
    sentence title
endform

sound = Read from file: wav$
textgrid = Read from file: tg$
ntiers = Get number of tiers
if tmax <= tmin
    selectObject: sound
    tmax = Get end time
endif
selectObject: sound
part = Extract part: tmin, tmax, "rectangular", 1, "yes"
spectrogram = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"

# layout in inches. Praat's TextGrid drawing splits its inner viewport into (ntiers + 4) equal rows
# and uses the bottom ntiers rows for the tiers; margins are 0.4 inch above and below.
left = 1.3
right = 11.3
tier_h = 0.5
top = 1.4
bottom = top + 0.8 + tier_h * (ntiers + 4)
tiers_top = bottom - 0.4 - ntiers * tier_h

Erase all
Font size: 10
Line width: 1

Select outer viewport: left, right, 0, 0.5
Text top: "no", title$

# waveform
Select outer viewport: left, right, 0.2, 1.7
selectObject: part
Draw: tmin, tmax, 0, 0, "no", "Curve"

# spectrogram (0-5 kHz), ending where the tiers begin
Select outer viewport: left, right, top, tiers_top + 0.4
selectObject: spectrogram
Paint: tmin, tmax, 0, 5000, 100, "yes", 50, 6, 0, "no"
Marks left every: 1, 1000, "yes", "yes", "no"
Text left: "yes", "Hz"

# tiers with boundaries through the spectrogram, plus time axis
Select outer viewport: left, right, top, bottom
selectObject: textgrid
Draw: tmin, tmax, "yes", "yes", "yes"

# tier names in the left margin
Font size: 9
for i to ntiers
    selectObject: textgrid
    name$ = Get tier name: i
    name$ = replace$ (name$, "_", " ", 0)
    y = tiers_top + (i - 0.5) * tier_h
    Select inner viewport: 0.05, left + 0.5, y - tier_h / 2, y + tier_h / 2
    Axes: 0, 1, 0, 1
    Text: 0, "left", 0.5, "half", name$
endfor

Select outer viewport: 0, right, 0, bottom
Save as 300-dpi PNG file: png$
