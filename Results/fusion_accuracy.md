# Fusion pipeline: test results

Test set: 30 words never seen in training, both speakers, all 7 emotions (420 files). Chance level: 14.3%. Silhouette: how well the emotions form separate clusters in the block output (-1 to 1, higher is better).

| Variant | Test accuracy (mean ± std, 5 seeds) | Macro F1 | Validation accuracy | Silhouette |
|---|---|---|---|---|
| fusion_concat | 99.7% ± 0.2 | 0.997 | 100.0% | 0.86 |
| fusion_gated | 99.7% ± 0.3 | 0.997 | 100.0% | 0.92 |

Recall per emotion (share of each emotion's test files classified correctly, all 5 seeds):

| Variant | angry | disgust | fear | happy | neutral | surprise | sad |
|---|---|---|---|---|---|---|---|
| fusion_concat | 100% | 100% | 100% | 100% | 100% | 99% | 100% |
| fusion_gated | 100% | 100% | 100% | 100% | 100% | 99% | 99% |
