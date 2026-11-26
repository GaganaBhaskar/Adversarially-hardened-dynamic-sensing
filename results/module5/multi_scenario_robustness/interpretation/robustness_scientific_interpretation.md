# AHDS Module 5 — Multi-scenario Robustness

## Experimental scope

The evaluation comprises 20 synthetic scenario configurations,
26 trials, five defensive strategies and 7,950 strategy-frame
evaluations. Module 3 adaptation remains disabled.

## Protection

DADRC reduces mean active censoring relative to fixed exposure
in 17 of 20 disturbance trials and ties in three.

Relative to probed exposure, DADRC ties in 19 trials and
has higher censoring in one. Universal protection superiority
is therefore not supported.

## Persistent disturbance recognition

DADRC detects 386 of 410 physically active frames (94.15%),
including 327 of 350 uncensored active frames (93.43%).

Four active frames are missed and 20 have unknown evidence.

No positive evidence occurs in the 1,120 nonactive frames
included in the ordinary detection evaluation. However,
20 of 60 separately evaluated benign illumination event
frames produce positive evidence.

Physical disturbance recognition does not establish intent.

## Recovery

Under the canonical criterion, DADRC restores 18 of 21
episodes. Probed exposure restores three and state-aware
exposure restores 18.

Among restored episodes, mean recovery delay is
4.61 frames for DADRC and 6.17 for state-aware exposure,
a descriptive reduction of 25.23%.

The fixed-exposure baseline is not a defensive recovery
mechanism because its exposure does not change.

## Information preservation

Protection, reference distortion, spatial gradient error,
quantization and matched-valid radiance reconstruction
must be interpreted jointly.

Detailed trial-paired metric comparisons are stored in
robustness_information_comparisons.csv.

No weighted composite superiority score is used.

## Noise-aware recovery

strategy  episodes  canonical_restored  noise_aware_restored  noise_aware_mean_delay
      S0         6                   3                     6                     0.0
      S1         6                   0                     0                     NaN
      S2         6                   0                     0                     NaN
      S3         6                   3                     6                     5.5
      S4         6                   3                     6                     5.5

## Adverse results

Protection losses, missed recognition, unknown evidence,
benign-event responses and recovery failure diagnostics
are preserved in robustness_failure_diagnostics.json.

No controller parameters were tuned after observing results.

## Scientific limitations

The evaluation uses synthetic scenes, fixed gain,
limited noise seeds and temporally dependent frames.
It does not establish population-level statistical inference,
real-camera robustness, malicious intent recognition,
downstream perception performance or embedded latency.

## Conclusion

The results support DADRC as a physical-layer defensive
controller combining strong censoring protection,
persistent disturbance recognition and controlled recovery.

They do not establish universal superiority across
all metrics or operating conditions.
