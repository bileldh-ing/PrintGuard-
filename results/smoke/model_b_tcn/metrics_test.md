### Model B (tcn) vs Model A baseline - split `test`
15 sequence variants, 513 frames. Alarm rule: 3-of-5 frames above 0.5. One evaluated frame = 2 printed layer(s); per-hour figures assume 60 s per evaluated frame.

| Class | Model | Frame AP | P | R | Detected seq. | Median delay (frames) | False alarms /100 frames | /print-hour* |
|---|---|---|---|---|---|---|---|---|
| stringing | A (frame) | 0.385 | 0.319 | 0.859 | 3/3 | 2.0 | 4.07 | 2.44 |
| stringing | B (temporal) | 0.965 | 0.654 | 1.000 | 3/3 | 2.0 | 0.81 | 0.49 |
| layer_shift | A (frame) | 0.334 | 0.303 | 0.355 | 4/8 | 0.0 | 1.88 | 1.13 |
| layer_shift | B (temporal) | 0.485 | 0.529 | 0.383 | 4/8 | 3.5 | 1.08 | 0.65 |

\* assumes 60 s per evaluated frame; the dataset does not publish layer times.
