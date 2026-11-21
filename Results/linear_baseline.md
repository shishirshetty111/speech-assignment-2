# Linear baseline: time-averaged WavLM -> logistic regression

C chosen on the validation words of the training speaker(s) only. Deterministic (no seeds).

On the training speaker's validation words every C reaches about 100%, so the tie goes to
the smallest C. The last column shows the test accuracy for every C in [0.001, 0.01, 0.1, 1.0, 10.0].

| Test setting | Layer 9 test accuracy | C | Range over all C |
|---|---|---|---|
| both speakers | 99.8% | 0.001 | 99.8-100.0% |
| OAF->YAF | 74.8% | 0.001 | 73.8-74.8% |
| YAF->OAF | 65.7% | 0.001 | 65.7-69.0% |

Sensitivity check, cross-speaker test accuracy per layer (not used to choose anything):

| Layer | OAF->YAF | YAF->OAF | Mean |
|---|---|---|---|
| 0 | 33.3% | 52.4% | 42.9% |
| 1 | 33.8% | 63.8% | 48.8% |
| 2 | 47.1% | 55.7% | 51.4% |
| 3 | 57.1% | 56.7% | 56.9% |
| 4 | 62.9% | 61.9% | 62.4% |
| 5 | 61.4% | 62.4% | 61.9% |
| 6 | 65.2% | 58.6% | 61.9% |
| 7 | 71.0% | 66.7% | 68.8% |
| 8 | 73.8% | 65.7% | 69.8% |
| 9 | 74.8% | 65.7% | 70.2% |
| 10 | 72.9% | 61.9% | 67.4% |
| 11 | 67.6% | 61.4% | 64.5% |
| 12 | 59.5% | 57.6% | 58.6% |
