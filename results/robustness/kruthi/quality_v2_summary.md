# Quality Score V2 test comparison

Research and educational use only; not a diagnostic result.

Approved test images: 874. Rows analyzed: 31464. The classifier was not rerun.

The supplied sweep used condition-specific hash seeds, which differs from the manual's `42 + image index` rule. This deviation is recorded for review.

| Corruption | Rows | Old AUC (95% CI) | V2 AUC (95% CI) | V2 - old |
|---|---:|---:|---:|---:|
| overall | 31464 | 0.553 (0.537–0.566) | 0.713 (0.696–0.728) | 0.160 |
| blur | 4370 | 0.405 (0.381–0.428) | 0.843 (0.826–0.859) | 0.438 |
| noise | 4370 | 0.538 (0.478–0.597) | 0.507 (0.453–0.557) | -0.031 |
| contrast | 4370 | 0.633 (0.605–0.657) | 0.757 (0.728–0.786) | 0.124 |
| brightness_up | 4370 | 0.542 (0.500–0.590) | 0.533 (0.488–0.576) | -0.009 |
| brightness_down | 4370 | 0.596 (0.543–0.647) | 0.459 (0.424–0.498) | -0.137 |
| jpeg | 4370 | 0.503 (0.443–0.562) | 0.515 (0.468–0.563) | 0.012 |
| downscale | 4370 | 0.552 (0.494–0.602) | 0.517 (0.461–0.574) | -0.035 |
