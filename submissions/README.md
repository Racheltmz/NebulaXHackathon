# Submissions

| Folder | What it is |
|---|---|
| `no_hard_label_retraining/predictions.zip` | models fitted on all labelled data, predicting the held-out set (public-test scores: Door 0.9474, ACV 0.375, Rail 0.8080, SHM 0.9322) |
| `with_hard_label_retraining/predictions.zip` | the same models retrained on all labelled data + confident held-out items (not yet scored) |
| `personal_check_ACV_hardcoded/predictions_NO_hard_label_retraining_ACV_HARDCODED.zip` | **NOT a model result**: the no-retraining zip with the ACV ranking overwritten so the faulty car implied by the public ACV score (car 01) is first. Door / Rail / SHM inside are the genuine no-retraining predictions. For a personal check only — do not submit. |

Rename the zip you upload to `predictions.zip`. Each contains `door_predictions.csv`, `acv_predictions.csv`, `rail_predictions.csv`, `shm_predictions.csv` at the top level.
