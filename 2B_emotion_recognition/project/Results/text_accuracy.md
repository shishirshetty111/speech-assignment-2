# Text pipeline: test results

Test set: 30 words never seen in training, both speakers, all 7 emotions (420 files). Chance level: 14.3%. Silhouette: how well the emotions form separate clusters in the block output (-1 to 1, higher is better).

| Variant | Test accuracy (mean ± std, 5 seeds) | Macro F1 | Validation accuracy | Silhouette |
|---|---|---|---|---|
| text_bert | 14.3% ± 0.0 | 0.036 | 14.3% | -0.02 |

Recall per emotion (share of each emotion's test files classified correctly, all 5 seeds):

| Variant | angry | disgust | fear | happy | neutral | surprise | sad |
|---|---|---|---|---|---|---|---|
| text_bert | 0% | 20% | 20% | 40% | 20% | 0% | 0% |
