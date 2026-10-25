# voice-spoof-eval

**An evaluation harness for detecting AI-generated speech in VoIP calls.**

> A 56-case protocol, a 4-point rubric, and a metrics pipeline for answering one question:
> *is this audio safe to trust?*

[![tests](https://github.com/EdemBey/voice-spoof-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/EdemBey/voice-spoof-eval/actions/workflows/ci.yml)

Built for a Computer Science project module at Roskilde University (F2026).

---

## The problem

Voice cloning needs seconds of audio. Voice authentication gates bank accounts, wallets and
device unlock. If a clone passes verification, the damage is done before a human ever listens.
Most detection work reports a single number: accuracy, or EER. That number hides the thing
that actually matters operationally — **the conditions under which the detector stops working**.

This repository is the measurement layer. It defines what "detected" means, runs the same
protocol against every platform, and reports failures by category instead of averaging them away.

## The protocol

| Element | Choice |
| --- | --- |
| Test set | 56 voice commands, 7 functional categories |
| Scale | 4-point ordinal rubric (0 = no response, 3 = correct and complete) |
| Repeats | up to 3 attempts per case; best score recorded, 0 if all attempts fail |
| Grading | ≥2.7 = A, ≥2.0 = B, ≥1.0 = C, <1.0 = F |
| Environment | controlled indoor, fixed 30 cm distance, default device sensitivity |
| Target | Apple Siri (iOS 17) and Google Assistant as surrogates for banking voice agents |

Categories: Activation, Information Retrieval, Device Control, Communication, Media,
Security & Privacy, Robustness.

The ordinal rubric is deliberate. A binary "spoofed / not spoofed" label collapses three very
different outcomes — silence, misparsing, and a *confidently wrong* action — into one bucket.
The wrong-but-confident case is the dangerous one, so it gets its own score.

## Results (full 56-case set)

Average score per category, 0–3 scale:

| Category | Siri | Google Assistant |
| --- | ---: | ---: |
| Activation | 2.25 | 0.00 |
| Information Retrieval | 1.25 | 2.38 |
| Device Control | 1.80 | 2.30 |
| Communication | 1.50 | 2.25 |
| Media | 3.00 | 1.20 |
| Security & Privacy | 2.88 | 1.88 |
| Robustness | 1.67 | 1.22 |
| **Overall** | **1.98** | **1.64** |
| **Grade** | **C** | **C** |

Reproduce the category aggregates with `python -m src.cli score`.

## Where it breaks

These are the findings worth arguing about in a review:

1. **Wake-word filtering is doing most of the defensive work.** Google Assistant scored 0.00 on
   Activation — eight out of eight clones rejected — yet 2.38 on Information Retrieval once the
   clone was manually triggered. The barrier is the gate, not the speaker model. Any workflow
   where activation is not required inherits almost none of that protection.
2. **Lossy codec compression does not protect the biometric.** After an Opus/WhatsApp-style
   round trip, the converted voice still matched the enrolled speaker at **0.85 cosine
   similarity** (other models stayed above 0.62). Compression removes the evidence without
   removing the vulnerability.
3. **Audible is not periodic.** Waveforms that sound clean to a listener returned
   **0 voiced frames** from pitch tracking: the RVC pipeline smears phase and harmonics just
   enough to destroy periodicity while leaving intelligibility intact. A human listener is
   insensitive to phase; a pitch tracker is not. That gap is a detection vector.
4. **Octave-leap errors are quantifiable.** One model produced a mean F0 of **411 Hz** on male
   source audio — an entire octave above the target's plausible range.
5. **Duration changes the verdict.** Short clips (wake-word only) were insufficient for either a
   response or a stable decision. Detection confidence is a function of how much audio you get,
   which is a latency budget problem, not just an accuracy problem.

## What is in this repository

```
data/
  rubric.csv            33 individually documented test cases (id, category, command, success criteria)
  results.csv           observed scores per platform for those 33 cases
  category_summary.csv  exact 56-case score distribution per category (from the project report)
src/
  rubric.py             rubric loading, aggregation, grading, markdown reporting
  metrics.py            F0 extraction, Jensen-Shannon divergence, DTW, cosine, spectral features
  failure_modes.py      machine-readable taxonomy of the failure modes found above
  cli.py                analyse / demo / score
tests/                  unit tests, offline, no audio required
```

**Scope note:** all 56 cases were run in the project; 33 of them are documented here at
individual-case level, and the remaining 23 (Device Control, Communication, Media) are recorded
as category-level score distributions in `category_summary.csv`. Both are included so the totals
stay verifiable. Nothing here is reconstructed or guessed.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.cli score          # aggregate the documented results and print grades
python -m src.cli demo           # octave-leap scenario
python -m src.cli demo --mode smear   # harmonic-smearing scenario
python -m src.cli analyse reference.wav converted.wav
python -m unittest discover -s tests -t . -v
```

`demo` needs no audio files and no model weights — it generates a harmonic reference signal and
a degraded "converted" version, either pitch-shifted by an octave (`--mode octave`) or with
decorrelated harmonic phases (`--mode smear`), so the metrics can be inspected end to end.

## Metrics

| Metric | What it answers |
| --- | --- |
| `f0_autocorrelation` | Is the signal periodic, and at what pitch? Returns NaN per unvoiced frame. |
| `voiced_ratio` | What share of frames even have a pitch? Collapses to 0 for smeared vocoder output. |
| `f0_distribution` + `js_divergence` | How far did the pitch distribution move from the reference? |
| `dtw_distance` | Did the pitch *contour* move together over time, or only the histogram match? |
| `cosine_similarity` | Do two embeddings point the same way (speaker identity proxy)? |
| `high_freq_ratio` | Is there a natural noise floor above 6 kHz, or was it synthesised away? |
| `spectral_flatness` | Is the spectrum noise-like (real room) or tonal (vocoder)? |

No deep model is required to get a signal. The cheap acoustic features already separate the
classes in the demo; embeddings are what tell you whether a *specific* identity was cloned.

## Limitations

- **Surrogate targets.** Banking-grade voice authentication was not accessible. Consumer
  assistants were used as a structurally similar proxy; results are a lower bound on the threat.
- **Single acoustic environment.** Indoor, fixed distance, default sensitivity. No crowdsourced
  device diversity, which is where published detectors usually fall over.
- **Offline vs real-time.** Offline conversion quality exceeded the real-time pipeline. Latency
  numbers were not formally measured, so nothing here should be read as a real-time claim.
- **33/56 at case level.** See the scope note above.
- **No adversarial arm.** Perturbation attacks (e.g. Malacopula) were reviewed but not run, so the
  detector is not evaluated against an attacker who optimises against it.

## My contribution

The evaluation design, the 56-case protocol and rubric, the analysis pipeline, the failure-mode
taxonomy, and the Python CLI/adapter layer (~950 lines) around the inference engine.
The core neural voice-conversion engine was provided by the project supervisor and is not
included here; this repository covers the measurement side, which was my part.

## Licence

MIT.
