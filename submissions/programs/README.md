# Programs used in the two final submissions

Each file is the exact code that was fitted on ALL labelled data and predicted the held-out set (`scripts/build_two_submissions.py` loads them by archive id; these copies are byte-exact).

| Task | File | Archive id | md5 (8) | all-data out-of-fold score | Input |
|---|---|---|---|---:|---|
| Door | `door_final.py` | `classical:41962a` | `53674f2b` | 1.0000 | input: raw [C,T] segments |
| ACV | `acv_final.py` | `classical_nx:seed` | `2aa9a21a` | 1.0000 | input: per-signal case-centred car vector (data/build_acv_nx_arrays.py) |
| Rail | `rail_final.py` | `classical_nx:68a6b9` | `eb748ac2` | 0.8524 | input: per-recording feature vector (data/build_nx_arrays.py) |
| SHM | `shm_final.py` | `classical_nx:46efff` | `022c1881` | 0.9241 | input: per-recording feature vector (data/build_nx_arrays.py) |

Note: `rail/classical_nx.py` (from `scripts/collect_best.py`) is the program with the highest *evolution* score, which is a different program (`43d41e`) from the one used here (`68a6b9`, higher out-of-fold score on all the data).
