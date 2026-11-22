# Data summary

Files found: 2800 unique recordings (2800 exact duplicate copies in the archive were skipped).

| | Count |
|---|---|
| Speakers | OAF (1400), YAF (1400) |
| Emotions | 7 |
| Words | 200 |
| Words recorded by both speakers in all 7 emotions | 200 |
| Original sample rates | 24414 Hz (2799), 96000 Hz (1) |
| Length before trimming (s) | min 1.25, median 2.05, max 2.98 |
| Length after trimming (s) | min 1.14, median 2.02, max 2.98 |
| Total speech after trimming | 1.57 h |

| Split | Words | Utterances |
|---|---|---|
| train | 140 | 1960 |
| val | 30 | 420 |
| test | 30 | 420 |

Utterances per emotion: angry 400, disgust 400, fear 400, happy 400, neutral 400, pleasant_surprise 400, sad 400

Irregular files:

- OA_bite_neutral.wav: speaker taken from the folder name (OAF)
- OAF_food_fear.wav: recorded at 96000 Hz (others 24414 Hz), resampled like the rest
