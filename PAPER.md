# Formula 1 Race Strategy Modeling

### Tyre degradation, pit timing, and a model that could not beat the grid

**Mahyar Sharafi Laleh** | September 2026
Code: [`github.com/Moonyar/f1-race-strategy`](https://github.com/Moonyar/f1-race-strategy) | Repository README: [`README.md`](README.md)

---

## Abstract

Formula 1 race strategy turns on a single quantity: how fast a tyre loses pace as it ages.
This paper estimates that quantity per compound from 53,708 green-flag racing laps across
58 dry races of the 2022 to 2024 ground-effect era, using FastF1 timing, telemetry and
weather data, after correcting each lap for fuel burn and track evolution.

The headline estimate is that **SOFT compounds lose 0.0398 s/lap of tyre age (95% CI 0.0222
to 0.0574)**, roughly twice the rate of MEDIUM (0.0189) and HARD (0.0228). Only the
SOFT-versus-rest gap is statistically defensible; **MEDIUM and HARD do not separate**
(difference -0.0039, p = 0.34) and their order flips with the tyre-age window.

Two models are built on top. A pit-window classifier predicting whether a driver pits within
three laps reaches **ROC-AUC 0.8098 on a held-out 2024 season** against a tyre-age-only
baseline of 0.5854. A finishing-position regressor **fails to beat the trivial baseline of
predicting the grid position** (MAE 2.756 against 2.718).

The most useful part of the project was a specification error caught by a built-in check
rather than by a metric, described in Section 5.2.

---

## 1. Motivation

Every pit-stop decision in Formula 1 is a trade between a known cost and an uncertain
benefit. A stop costs roughly 20 seconds. Staying out costs whatever pace the current tyres
are still bleeding away. A strategist who cannot quantify the second number cannot evaluate
the first.

That makes tyre degradation the natural quantity to estimate. It is also easy to sanity
check: a degradation slope is a number in seconds per lap that anyone who follows the
sport can look at and call wrong. Physics demands that softer compounds degrade faster. If the model says otherwise,
the model is broken.

I set out to build three things: the degradation curves, a model of when teams actually pit,
and a model of where cars finish. The first two worked. The third did not.

---

## 2. Data

| | |
| --- | --- |
| Source | FastF1 API (official F1 timing feed), cached locally |
| Seasons | 2022 to 2024 |
| Races ingested | 68 |
| Races in modelling set | 58 (dry only) |
| Driver-laps ingested | 74,603 |
| Driver-laps after cleaning | 53,708 (72.0% retained) |
| Stints used for degradation | 2,717 |
| Telemetry coverage | 100.0% of modelling laps |
| Drivers / circuits | 28 / 25 |

Three streams are joined into one driver-lap table: per-lap timing (compound, tyre age,
stint, position, pit markers, track status), per-sample car telemetry aggregated to the lap
(max speed, mean throttle, brake percentage, DRS percentage, max RPM, mean gear), and
trackside weather joined on nearest timestamp to each lap's *start* time.

### 2.1 Why only three seasons

2022 is the first year of the ground-effect regulations, which changed the cars'
aerodynamics, minimum weight, and most relevantly moved Formula 1 from 13-inch to 18-inch
wheels with new Pirelli constructions. **Tyre degradation before and after that change is not
the same physical process.** Pooling 2019 to 2021 with 2022 onward would be mixing two
regimes to get a bigger sample.

FastF1 timing data only goes back to 2018 in any case, so the alternative was never a long
history.

### 2.2 Cleaning

| Filter | Laps remaining | Dropped | % of raw |
| --- | --- | --- | --- |
| Raw driver-laps ingested | 74,603 | 0 | 100.0 |
| Have a timed lap | 72,690 | 1,913 | 97.4 |
| Drop in-laps | 70,322 | 2,368 | 94.3 |
| Drop out-laps | 68,170 | 2,152 | 91.4 |
| Drop SC / VSC / red-flag laps | 64,904 | 3,266 | 87.0 |
| `IsAccurate` backstop | 64,479 | 425 | 86.4 |
| Dry slick compounds only | 60,366 | 4,113 | 80.9 |
| Drop races with logged rainfall | 53,832 | 6,534 | 72.2 |
| Have tyre age | 53,755 | 77 | 72.1 |
| Within [0.9, 1.2]x session median | 53,708 | 47 | 72.0 |

Losing 28% of raw laps is expected rather than alarming. In-laps and out-laps run 20 or more
seconds off the pace and would destroy any degradation fit. Safety-car laps are not racing
laps at all.

FastF1's own `IsAccurate` flag already implies
non-pit and green-flag, so it subsumes the explicit filters above it. The pipeline applies
the explicit filters anyway so the cleaning does not depend on an undocumented flag
definition that could change between versions. An assertion after cleaning confirms no
in-lap, out-lap or non-green lap survived.

---

## 3. Corrections

Both corrections exist for the same reason: without them, effects that have nothing to do
with the tyre get absorbed into the tyre-age slope and bias it toward zero.

### 3.1 Fuel

Cars start heavy and get lighter, so raw lap times fall through a stint even while tyres
degrade.

```
fuel_remaining_kg  = 100 * (total_laps - lap_number) / total_laps
lap_time_corrected = lap_time - fuel_remaining_kg * 0.03
```

`fuel_start_kg = 100` (the regulation maximum is 110 kg and teams start near but under it),
and `seconds_per_kg = 0.03` (the common rule of thumb is 0.03 to 0.035, and it is
circuit-dependent).

**These are assumptions, not measurements, and their sensitivity is untested.** A linear burn
model also ignores fuel-saving and lift-and-coast. The direction of the bias is knowable
though: a larger `seconds_per_kg` steepens every compound's slope roughly equally, so the
*ordering* and the *gaps between compounds* are considerably more robust than the absolute
numbers. The comparative claims in Section 5 survive the assumption; the exact values depend
on it.

### 3.2 Track evolution

A circuit rubbers in and the whole field gets faster, independently of tyres and fuel.

Per race, I take the field's median fuel-corrected lap time as a function of lap number,
smooth it with a centred 5-lap rolling median, and carry it as an explicit regressor.

Because that term is constructed in seconds of field-wide pace, **its fitted coefficient must
land near -1**. That makes it a built-in check on the specification, and it turned out to
be the most useful diagnostic in the project.

---

## 4. Method

**Degradation.** Fuel-corrected lap time regressed on tyre age interacted with compound,
with **stint fixed effects** (every stint gets its own intercept) and standard errors
**clustered by race**, since laps within one race share weather, surface and a single
safety-car history and are not independent.

**Pit-window classifier.** Given the state at lap L, does this driver pit within the next
three laps? `HistGradientBoostingClassifier`. Features are strictly what a strategist knows
at lap L: tyre age and compound, pace versus the field baseline, a three-lap pace trend,
position, laps remaining, stops so far, track temperature and circuit. Stint *total* length
is deliberately absent, because at lap L nobody knows how long the stint will be, and that is
precisely the answer.

**Finishing position.** Both variants predict the **delta from grid** and add grid position
back, rather than predicting position outright. Grid position dominates in Formula 1, so
the model only has to learn the change from it.

**Splits are by season, never random.** Train 2022, validate 2023, test 2024. No
`train_test_split` appears anywhere in the pipeline.

---

## 5. Results

### 5.1 Degradation slopes

| Compound | s/lap | Std. err. | 95% CI | t | Laps |
| --- | --- | --- | --- | --- | --- |
| SOFT | **0.0398** | 0.0090 | 0.0222 to 0.0574 | 4.44 | 5,661 |
| MEDIUM | 0.0189 | 0.0046 | 0.0099 to 0.0280 | 4.10 | 19,606 |
| HARD | 0.0228 | 0.0034 | 0.0161 to 0.0295 | 6.64 | 28,123 |

SOFT degrades fastest, which is the ordering physics demands, and it holds in every
tyre-age window tested (15, 20, 25, 30 laps, and unrestricted).

### 5.2 A specification error that the fit statistics did not show

The obvious specification is a pooled OLS with circuit and driver fixed effects. It produces
plausible-looking slopes and a reasonable fit. **It is also biased.**

The tell was the track-evolution coefficient. It is constructed in seconds of field-wide
pace, so it should come out near -1. In the pooled model it came out at **+0.397**, which
means the specification is wrong however good the fit looks.

The cause is **between-stint selection**. Stint length is a strategist's choice and differs
systematically by compound: HARD stints average 27 laps, MEDIUM 19, SOFT 16. Stints also
differ in traffic, car damage and race phase. A pooled regression identifies the tyre-age
slope partly from that between-stint variation, which is confounded with everything the
strategist was reacting to when they chose the stint length. It had also inverted the
MEDIUM/HARD ordering.

Absorbing a **stint fixed effect** fixes it. The track-evolution coefficient lands at
**-1.016**, as expected. Both models remain in the codebase.

The takeaway: it helps to have at least one term in the model whose correct value is known
in advance. Fit statistics say how well a model fits, not whether it is the right model.

### 5.3 What cannot be claimed

**MEDIUM and HARD do not separate.** The three slopes invite a ranking, so the ranking is
tested rather than eyeballed from overlapping intervals:

| Contrast | Difference | p | Significant |
| --- | --- | --- | --- |
| SOFT - MEDIUM | 0.0209 | 0.016 | yes |
| SOFT - HARD | 0.0170 | 0.050 | no |
| MEDIUM - HARD | -0.0039 | 0.338 | no |

Only the SOFT-versus-rest gap is claimed. Three explanations are plausible and this data
cannot distinguish between them: Pirelli brings a different C1 to C5 triplet to each circuit,
so "HARD" is not one physical tyre across races; HARD runs in much longer stints, and a
linear slope over a longer age range is not comparable to one over a shorter range; or the
compounds are adjacent and the true gap is small against race-to-race noise.

**The cliff is not measurable here, and the curvature has the wrong sign.** A quadratic term
is significant for MEDIUM and HARD, but **negative**: pace loss decelerates with tyre age.
That is the opposite of the cliff I went looking for, and it is almost certainly selection
rather than physics. Teams pit as soon as a tyre falls away, so the stints surviving to high
tyre age are exactly the ones where the tyre was still working. The post-cliff laps that
would produce positive curvature are absent by construction. A survival model on stint length
is the right tool and this is not it.

### 5.4 Pit-window classifier

| Split | Laps | Base rate | ROC-AUC | PR-AUC | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Validation (2023) | 18,066 | 0.1021 | 0.8002 | 0.3458 | 0.312 | 0.556 |
| **Test (2024)** | 19,530 | 0.0885 | **0.8098** | **0.3028** | 0.240 | 0.685 |

**The base rate is 8.8%, so accuracy is not useful here**: a model predicting "never pits"
scores 91.1%. PR-AUC against the base rate is the better measure, and the baseline is not a
coin flip but **ranking laps by tyre age alone**, which already
achieves ROC-AUC 0.5854. The model's contribution over that is **+0.2244 ROC-AUC**.

The decision threshold is chosen on the validation season and carried to test unchanged.

### 5.5 Finishing position: does not beat the baseline

| Model | Test MAE | Grid baseline | Beats grid? | R² |
| --- | --- | --- | --- | --- |
| Post-race summary | 2.756 | 2.718 | **no** (-0.039) | 0.519 |
| Strict pre-race | 3.171 | 2.718 | **no** (-0.453) | 0.454 |

**Neither variant beats the trivial baseline of predicting the grid position on the held-out
2024 season.**

Three things about how it failed:

1. **It is level, not beaten.** The post-race model sits 0.039 of a position behind the
   baseline across 248 driver-races. It does not add anything over knowing the grid; it is
   not wildly wrong.
2. **It beat the baseline on validation and not on test.** On 2023 it won by more than a
   position (2.372 against 3.518). The baseline itself is far stronger in 2024 (2.718) than
   in 2023 (3.518), because 2024 was a more predictable season in which cars finished closer
   to where they qualified. A model trained on earlier seasons learned a world with more
   position shuffling than the one it was tested on. A random split would have hidden this;
   the time-based split shows it.
3. **The training set is tiny.** Roughly 400 driver-races is very little for this target.

The formulation and hyperparameters were fixed on validation before 2024 was evaluated
**once**. No re-tuning happened after seeing the test number. Grid position is enormously
predictive in Formula 1 because overtaking is hard, so failing to beat it is the normal
outcome rather than a bug.

---

## 6. Validation and leakage

`src/leakage_check.py` verifies each item mechanically, parsing the pipeline's AST rather
than grepping text, recomputing statistics from saved artefacts, and proving the fitted
imputer learned the training-fold median rather than the all-data median. Items that cannot
be proven by code print as MANUAL with their reasoning, so "verified" and "argued" stay
distinguishable. Result: **8 PASS, 1 MANUAL, 0 FAIL.**

**One caveat.** The field-baseline lap time is a
within-race aggregate over contemporaneous laps by other drivers, smoothed with a **centred**
window. Two consequences, both accepted knowingly:

- It uses other drivers' current laps. That is defensible, since it is exactly what a pit wall
  sees live.
- The centred window means lap L's baseline also uses laps L+1 and L+2 of the same race. That
  is a genuine two-lap lookahead in a field-wide aggregate. It is not driver-specific and
  carries no information about whether *this* driver pits, but it is the one place the
  pipeline is not strictly causal. A trailing window would remove it at the cost of a laggier
  baseline.

A smaller one: the post-race finishing model's `deg_exposure` feature uses degradation slopes
fitted on all three seasons, including the test season. It is three pooled constants and the
model does not beat its baseline regardless, but a stricter pipeline would fit them on
training seasons only.

---

## 7. Limitations

- **Fuel constants are assumed, not fitted**, and their sensitivity is untested.
- **Wet races are excluded entirely**, so nothing here speaks to wet strategy, which is where
  races are actually won.
- **Degradation is linear in tyre age** within the observed range; curvature came out negative
  and the cliff is unmeasurable in this data.
- **No traffic or gap features.** Whether a driver is stuck behind someone is a first-order
  input to real pit timing, since the undercut is the entire game, and the model has no idea.
  **This is the single biggest gap.**
- **Degradation slopes are pooled across drivers**, so a driver who manages tyres well is
  averaged with one who does not. Driver fixed effects absorb the level, not the slope.
- **Survivorship:** a driver who retires early contributes only their early laps, which are
  their freshest tyres.
- **MEDIUM and HARD are not statistically separable.**
- **The finishing-position model does not beat its baseline.**

---

## 8. What I would do differently

- **Add traffic and gap features.** Gap to the car ahead and behind, and time spent within
  DRS range, are what pit timing actually responds to. This is the highest-value addition and
  the pipeline has the telemetry to build it.
- **Fit the fuel coefficient instead of assuming it.** It is identifiable from within-stint
  pace on long runs, and it would turn the largest assumption in the paper into an estimate.
- **Model stint length as survival.** A Cox model on stint duration with tyre age as the time
  axis is the correct tool for the cliff, and it handles the censoring that makes the
  quadratic term meaningless here.
- **Model wet races separately** rather than dropping them.
- **Hierarchical degradation by driver and circuit.** Partial pooling would let slopes vary by
  driver without fitting 28 independent noisy estimates.

---

## 9. Conclusion

Tyre degradation is estimable from public timing data to a useful precision, and the estimate
behaves the way physics requires: SOFT compounds lose about 0.04 s/lap of tyre age, roughly
double the harder compounds. MEDIUM and HARD cannot be told apart in this data, and saying so
is part of the result.

A pit-window classifier built on the same features reaches 0.81 ROC-AUC on a held-out season
against a 0.59 tyre-age baseline, which is a genuine contribution. A finishing-position model
does not beat the trivial grid baseline, which is the normal outcome in a sport where
overtaking is hard.

The most serious error in the project (Section 5.2) was a misspecification that produced
reasonable-looking output, and it was caught by a term whose correct value was known in
advance rather than by any fit statistic.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh          # ingest -> features -> degradation -> models -> checks
streamlit run app.py
```

Model selection history is in [`docs/model_selection.md`](docs/model_selection.md). Full
method notes and the repository map are in [`README.md`](README.md).
