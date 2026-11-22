# Speech pipeline: test results

Test set: 30 words never seen in training, both speakers, all 7 emotions (420 files). Chance level: 14.3%. Silhouette: how well the emotions form separate clusters in the block output (-1 to 1, higher is better).

| Variant | Test accuracy (mean ± std, 5 seeds) | Macro F1 | Validation accuracy | Silhouette |
|---|---|---|---|---|
| speech_wavlm | 99.8% ± 0.2 | 0.998 | 100.0% | 0.78 |
| speech_logmel | 99.7% ± 0.2 | 0.997 | 99.9% | 0.71 |

Recall per emotion (share of each emotion's test files classified correctly, all 5 seeds):

| Variant | angry | disgust | fear | happy | neutral | surprise | sad |
|---|---|---|---|---|---|---|---|
| speech_wavlm | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| speech_logmel | 100% | 99% | 100% | 99% | 100% | 99% | 100% |
