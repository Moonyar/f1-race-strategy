# Finishing-position model: how the formulation and hyperparameters were chosen

The test season (2024) was evaluated **exactly once**, after both decisions below
were final. It was not used to choose anything.

## 1. Formulation (chosen on principle, before any evaluation)

Predict the **delta from grid**, `final_position - grid_position`, and add the
grid position back, rather than predicting the finishing position outright.

Grid position is the dominant predictor of a finishing position in F1. Asking a
tree ensemble with ~400 training rows to predict the absolute position forces it
to relearn the near-identity grid-to-finish mapping from scratch, spending its
capacity on the part of the problem that is already known. Modelling the
residual gives it that structure for free and lets all of its capacity go to the
deviations, which is where the interesting signal is (race pace, number of
stops, attrition). This is ordinary offset/residual modelling.

## 2. Hyperparameters (chosen on the validation season only)

Fit on 2022, scored on 2023. Four configurations, both formulations:

| formulation | leaves | lr | l2 | valid MAE |
| --- | --- | --- | --- | --- |
| absolute | 15 | 0.05 | 1.0 | 2.440 |
| absolute | 7 | 0.05 | 1.0 | 2.500 |
| absolute | 7 | 0.03 | 5.0 | 2.328 |
| absolute | 31 | 0.03 | 0.5 | 2.411 |
| delta | 15 | 0.05 | 1.0 | 2.508 |
| delta | 7 | 0.05 | 1.0 | 2.418 |
| delta | 7 | 0.03 | 5.0 | 2.401 |
| delta | 31 | 0.03 | 0.5 | 2.496 |

Validation grid baseline MAE: 3.518.

`leaves=7, lr=0.03, l2=5.0` is the best configuration within each formulation,
so it was taken for both models.

**Why delta was kept anyway:** on the validation season the absolute formulation scored
marginally better than delta (2.328 vs 2.401). At n=191 driver-races that gap is
well inside noise, and the delta formulation had already been chosen on the
argument above, so it was kept. Switching formulation on a 0.07-position
validation difference would be fitting the validation set, and the pre-race
variant preferred delta anyway (3.412 vs 3.487).
