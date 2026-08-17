MEP Architecture V5.1: Formal Preregistration Document

Author: 3MOORE. BBB

Date: August 17, 2026

Status: Pre-Data Collection (Cryptographic Hash Pending)

Executive Summary

This document serves as the formal, cryptographically frozen preregistration for the experimental validation of the MEP Architecture's V5.1 memory substrate. Addressing previous methodological flaws, this protocol shifts the evaluation to a completely unseen benchmark (Split-CIFAR10), equalizes the memory budget against an Experience Replay baseline, tracks the harmonic mean of stability and plasticity, and enforces strict, prospective statistical reporting.

1. The Hypothesis & Mechanism

We hypothesize that Catastrophic Forgetting can be significantly mitigated by protecting the local functional geometry of a neural network via sampled directional derivatives, rather than constraining its internal parameter coordinates.

Using Hutchinson's trace estimator to calculate a Jacobian-Vector Product (JVP) over a strict budget of N anchor points, the operator applies an elastic tether. The absolute parameter coordinates are allowed to scramble to learn the new task, provided the directional derivatives around the anchor points remain invariant.

2. Experimental Design

Benchmark: Split-CIFAR10.
Task A: Classes 0-4 (5,000 training images per class).
Task B: Classes 5-9 (5,000 training images per class).
Network Architecture: Standard discrete SimpleCNN.
Strict Memory Budget: Both the Baseline and the MEP algorithm are granted an exact memory budget of  data points extracted from Task A.
Baseline Usage: 50 labeled (x, y) pairs stored in a Replay Buffer.
MEP Usage: 50 unlabeled (x) coordinates used to anchor the JVP operator.

3. Baselines & Seeds

Baseline: Experience Replay (ER). A standard continual learning baseline that interleaves samples from the 50-point Replay Buffer during Task B training.

Target Model: MEP Jacobian Tether (JVP penalty applied to the 50 anchor coordinates).
Seeds: 10 independent random initializations.

4. Metrics & Statistical Adjudication

Because perfect Task-A retention can be achieved by refusing to learn Task B (the Stability-Plasticity dilemma), the primary metric of success is the Harmonic Mean of Task A Retention Accuracy and Task B Acquisition Accuracy.

HarmonicMean = 2 x (Retention A x Acquisition B) / (Retention A + Acquisition B)

Statistical significance will be determined using Welch’s t-test (assuming unequal variances), reporting the -value, Cohen's  effect size, and 95% Confidence Intervals.

5. Explicit Failure Conditions (Falsifiability)

The MEP Jacobian Tethering hypothesis will be considered FAILED if:
Baseline Equivalence: The MEP architecture's Harmonic Mean fails to outperform the Experience Replay baseline by a statistically significant margin ($p \ge 0.05$). If storing 50 labels performs equally to or better than computing complex Jacobian-Vector Products on 50 coordinates, the added computational complexity of the JVP engine is unjustified.

Cryptographic Provenance

To eliminate post-hoc storytelling and ensure the executable artifact is tied to this protocol, the SHA-1 Git commit hash of the execution script (v5_1_cifar_crucible.py) is recorded below prior to data collection:
Git Commit Hash: [0dd6dc7d60861d78bdd533b791382036973ccce2]
