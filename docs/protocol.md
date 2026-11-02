# Experimental Protocol

The study evaluates selective evidence preservation rather than only anomaly
detection. The primary outcome is the relationship between evidence recall and
retained-video ratio.

Thresholds, context-policy weights, temporal buffer parameters, and calibration
methods must be selected on a validation partition only. The held-out test
partition must not be used to choose any configuration.

All metrics must be stored per video and per event before aggregation.
