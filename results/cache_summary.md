# Cache summary

Cache: `cache/` as uploaded, feature hash `647a1e76b0`. Measure: coh, 5 bands (paper 3.3), EC, 19 ch, interpolated missing. All integrity assertions passed (855 columns, no NaN, no constant column, aligned lengths, one `X_long` row per subject, labels and ages constant within subject and equal to the participants CSV, section counts equal to config).

## Subjects and sections per class

| index | subjects | sections | sections per subject (config) |
|---|---|---|---|
| female | 73 | 584 | 8 |
| male | 128 | 640 | 5 |

Total: 201 subjects, 1224 sections.

## Sections per subject

| sex | sections | subjects |
|---|---|---|
| female | 8 | 73 |
| male | 5 | 128 |

Available complete sections per recording in the per-subject cache: 13: 1, 14: 8, 15: 192 (sections: subjects).

## Age bins per class

| age_bin | female | male | All |
|---|---|---|---|
| 20-25 | 22 | 49 | 71 |
| 25-30 | 14 | 40 | 54 |
| 30-35 | 6 | 6 | 12 |
| 35-40 | 0 | 1 | 1 |
| 55-60 | 1 | 2 | 3 |
| 60-65 | 6 | 10 | 16 |
| 65-70 | 15 | 8 | 23 |
| 70-75 | 8 | 10 | 18 |
| 75-80 | 1 | 2 | 3 |
| All | 73 | 128 | 201 |

## Sex ratio within age clusters

Split at bin midpoint 45: young = bins 20–25 to 35–40, older = 55–60 to 75–80 (LEMON has no 40–55 bins).

| cluster | female | male | female_share |
|---|---|---|---|
| older | 31 | 32 | 0.492 |
| young | 42 | 96 | 0.304 |

## skipped.csv

| subject | reason |
|---|---|
| sub-032483 | error: ValueError: sfreq 100.0 != config 250.0 |
| sub-032484 | error: ValueError: sfreq 100.0 != config 250.0 |

## Interpolated channels (70 subjects)

Subjects whose missing channels were rebuilt by spherical-spline interpolation:

sub-032301 (T7), sub-032302 (T7), sub-032305 (T7), sub-032306 (T7 T8 O1), sub-032307 (F7 T8), sub-032310 (T7 T8), sub-032313 (Fp2 Cz T8), sub-032319 (T8), sub-032323 (T7), sub-032324 (Cz), sub-032325 (T7), sub-032326 (Fz T8), sub-032329 (Cz), sub-032333 (Cz), sub-032336 (Fp1 O1), sub-032345 (T7 T8), sub-032349 (T8), sub-032350 (T8), sub-032355 (Fp1), sub-032356 (T7 T8), sub-032357 (C4), sub-032360 (Fp1 F8), sub-032361 (T7), sub-032362 (Fp1), sub-032364 (T7 T8), sub-032367 (T7 T8), sub-032368 (Fp1), sub-032369 (T7 T8), sub-032370 (T7 T8), sub-032377 (T7), sub-032378 (T7 T8), sub-032381 (P7), sub-032383 (F8), sub-032387 (T7), sub-032392 (T7 T8), sub-032393 (T7 T8), sub-032394 (Cz), sub-032396 (F7), sub-032397 (Fz), sub-032399 (Fp1), sub-032402 (Fp1 F7 T7), sub-032403 (F7), sub-032409 (T7 T8), sub-032411 (Fp2), sub-032413 (Fp2), sub-032416 (Fp2 F7), sub-032428 (Fz T7 T8 P7), sub-032429 (Fp2), sub-032434 (P3), sub-032447 (T8), sub-032448 (F7), sub-032450 (T7 T8), sub-032457 (Cz), sub-032465 (Cz), sub-032466 (Fp1), sub-032468 (F7 F3 Cz), sub-032469 (Fz Cz), sub-032472 (F7), sub-032473 (T8), sub-032474 (F7), sub-032478 (Cz), sub-032479 (F7), sub-032491 (Cz), sub-032493 (T7), sub-032497 (Cz), sub-032498 (Fp2 F7), sub-032509 (Fp2), sub-032510 (T8), sub-032517 (T7), sub-032522 (T7 T8)

## Feature range

min 0.0196, max 1.0000. Columns reaching ≥ 0.999 in some section: 60.

| subject | sex | sections_hit | sections | pairs |
|---|---|---|---|---|
| sub-032334 | female | 8 | 8 | Fp2-F8 |
| sub-032342 | female | 8 | 8 | P7-O1 |
| sub-032347 | male | 5 | 5 | P7-O1 |
| sub-032373 | female | 8 | 8 | P3-O1 P7-O1 P7-P3 P8-O2 |
| sub-032380 | female | 8 | 8 | P4-P8 |
| sub-032388 | female | 5 | 8 | P7-P3 |
| sub-032437 | male | 5 | 5 | F4-F8 P3-Pz P7-P3 P7-Pz |
| sub-032478 | male | 5 | 5 | Fp2-F4 Fp2-Fz Fz-F4 |
| sub-032481 | female | 8 | 8 | F7-F3 |
| sub-032502 | male | 4 | 5 | Fp2-F4 |
| sub-032514 | male | 5 | 5 | Fp1-F7 |

Each of these subjects has one neighbouring channel pair (or a small cluster) at coherence ≈ 1 in most bands and sections: the signature of electrode bridging (two electrodes shorted by gel), not of interpolation (only one of them had an interpolated channel). They are kept; see notes/DECISIONS.md D5.
