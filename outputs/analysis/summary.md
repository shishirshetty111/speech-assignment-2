# Alignment analysis

Reference for agreement: arpa_g2p. Agreement is not accuracy (no hand-marked boundaries exist for this data).

## Experiments

| experiment | words | phones | spn_phones | pauses | pause_s | mean_phone_ms | phones_at_30ms | speech_loglik_f2b | speech_loglik_isle |
|---|---|---|---|---|---|---|---|---|---|
| arpa_adapted | 241 | 1063 | 0 | 34 | 7.67 | 75.8 | 82 | -46.13 | -47.39 |
| arpa_baseline | 241 | 1022 | 6 | 34 | 7.69 | 78.9 | 78 | -46.86 | -48.54 |
| arpa_g2p | 241 | 1063 | 0 | 34 | 7.67 | 75.8 | 82 | -46.98 | -48.54 |
| arpa_own_g2p | 241 | 1063 | 0 | 34 | 7.67 | 75.8 | 83 | -46.98 | -48.52 |
| mfa_g2p | 241 | 1030 | 0 | 32 | 7.66 | 82.4 | 120 | -45.51 | -45.84 |
| scratch | 241 | 1050 | 0 | 81 | 28.57 | 60.3 | 722 |  |  |

## Boundary agreement with arpa_g2p

| experiment | tier | matched | mean_ms | median_ms | within_20ms_pct | max_ms |
|---|---|---|---|---|---|---|
| arpa_adapted | words | 241/241 | 0.0 | 0.0 | 100.0 | 0 |
| arpa_adapted | phones | 1063/1063 | 0.1 | 0.0 | 100.0 | 20 |
| arpa_baseline | words | 241/241 | 0.7 | 0.0 | 99.2 | 20 |
| arpa_baseline | phones | 1016/1063 | 0.8 | 0.0 | 99.7 | 660 |
| arpa_own_g2p | words | 241/241 | 0.0 | 0.0 | 100.0 | 0 |
| arpa_own_g2p | phones | 1060/1063 | 0.0 | 0.0 | 100.0 | 10 |
| mfa_g2p | words | 241/241 | 25.7 | 10.0 | 84.4 | 2460 |
| scratch | words | 241/241 | 1521.9 | 1360.0 | 1.7 | 4750 |
| scratch | phones | 1012/1063 | 1568.3 | 1440.0 | 1.0 | 4800 |

## Phones at the 30 ms minimum (arpa_g2p)

82 phones (full list in short_phones.csv). Most common: T 15, N 14, AH0 8, R 6, L 5, AO1 4, D 4, EH1 3

## ISLE minimal pairs: vowel duration and formants (arpa_g2p; all experiments in isle_vowels.csv)

| experiment | file | median_F0_Hz | ceiling_Hz | word | phones | vowel | vowel_ms | F1_Hz | F2_Hz |
|---|---|---|---|---|---|---|---|---|---|
| arpa_g2p | _01 | 111 | 5000 | white | W AY1 T | AY1 | 150 | 546 | 1389 |
| arpa_g2p | _01 | 111 | 5000 | bait | B EY1 T | EY1 | 200 | 406 | 1899 |
| arpa_g2p | _02 | 115 | 5000 | new | N Y UW1 | UW1 | 240 | 291 | 1033 |
| arpa_g2p | _02 | 115 | 5000 | no | N OW1 | OW1 | 350 | 377 | 847 |
| arpa_g2p | _03 | 113 | 5000 | bad | B AE1 D | AE1 | 220 | 594 | 1767 |
| arpa_g2p | _03 | 113 | 5000 | bed | B EH1 D | EH1 | 180 | 527 | 1836 |
