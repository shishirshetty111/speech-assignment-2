# Cross-speaker test

Trained on one speaker, tested on the other speaker's recordings of the 30 test words (210 files per direction). Mean ± std over 5 seeds. Chance: 14.3%.

| Variant | OAF->YAF | YAF->OAF | Both directions | Macro F1 |
|---|---|---|---|---|
| speech_wavlm | 57.4% ± 10.0 | 45.5% ± 5.8 | 51.5% | 0.478 |
| speech_logmel | 41.3% ± 3.7 | 34.0% ± 4.9 | 37.7% | 0.342 |
| fusion_concat | 52.7% ± 8.5 | 38.0% ± 2.8 | 45.3% | 0.415 |
| fusion_gated | 56.6% ± 12.8 | 47.8% ± 8.0 | 52.2% | 0.497 |

Recall per emotion (both directions, 5 seeds):

| Variant | angry | disgust | fear | happy | neutral | surprise | sad |
|---|---|---|---|---|---|---|---|
| speech_wavlm | 25% | 72% | 71% | 20% | 96% | 21% | 55% |
| speech_logmel | 57% | 34% | 75% | 0% | 45% | 3% | 50% |
| fusion_concat | 20% | 62% | 72% | 7% | 97% | 24% | 35% |
| fusion_gated | 23% | 66% | 75% | 29% | 96% | 33% | 44% |
