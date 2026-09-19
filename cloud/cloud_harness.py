"""Cloud (NUS SoC SLURM cluster) equivalent of common/harness.py's
run_candidate_sandboxed. OpenEvolve itself still runs LOCALLY; only the
fit()/predict() execution for each CV fold / train-test split is shipped to
a GPU compute node via SLURM, since these tracks require real GPU deep
learning (torch+CUDA) that the local sandbox deliberately does not provide.

HONEST ANTI-CHEAT DISCLOSURE (read before trusting this like the local
Docker sandbox): the NUS SoC cluster is a shared academic cluster. Unlike
the local sandbox, jobs here run with ordinary outbound network access --
there is no `--network none` equivalent available to us on shared compute
nodes, so a network-capable candidate could in principle reach the
internet. What still holds unconditionally regardless of network access:
  - True labels for whatever a call is being scored on are NEVER uploaded to
    the cluster in the first place (only that fold's *training* labels are
    sent, which the model is allowed to see; y_test is computed against the
    returned predictions locally, in this process, never on the cluster).
  - predict() still structurally never receives labels.
  - The AST public-signature contract (common/contract.py) is still
    enforced locally before any upload happens.
  - Each call gets its own fresh, uniquely-named scratch directory on the
    cluster; nothing persists or is shared between calls or candidates.
So the worst case of a malicious candidate here is "wastes GPU time or
makes unwanted network calls," not "sees held-out answers" -- the
information-flow guarantee that actually matters is unchanged.
"""
from __future__ import annotations

import pickle
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SAAD304 = str(Path.home() / ".local" / "bin" / "saad304")
REMOTE_ROOT = "timesfm-classifier/ps3evolve_gpu_scratch"
POLL_INTERVAL_S = 5
DEFAULT_TIMEOUT_S = 600  # generous: queue time + model load + train + predict


class CloudCandidateError(Exception):
    pass


RETRY_BACKOFF_S = (2, 5, 10)  # 4 tracks share one saad304/VPN relay -- back off on contention


def _remote(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run([SAAD304, cmd], capture_output=True, text=True, timeout=timeout)


def _remote_best_effort(cmd: str, timeout: int = 60) -> None:
    # For cleanup/diagnostic calls only (rm -rf, scancel, log tail): a
    # transient timeout here under parallel contention must never crash an
    # otherwise-successful candidate evaluation, so swallow it rather than
    # letting subprocess.TimeoutExpired propagate uncaught.
    try:
        _remote(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _with_retries(fn, *args, attempts: int = len(RETRY_BACKOFF_S) + 1, **kwargs):
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            if i < len(RETRY_BACKOFF_S):
                time.sleep(RETRY_BACKOFF_S[i])
    raise CloudCandidateError(f"timed out after {attempts} attempts: {last_exc}")


def _remote_upload(local: str, remote: str, timeout: int = 120) -> None:
    def _do():
        r = subprocess.run([SAAD304, "upload", local, remote], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise CloudCandidateError(f"upload failed ({remote}): {r.stderr[-500:]}")
    _with_retries(_do)


def _remote_download(remote: str, local: str, timeout: int = 120) -> bool:
    def _do():
        r = subprocess.run([SAAD304, "download", remote, local], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and Path(local).exists()
    return _with_retries(_do)


def run_candidate_cloud(
    program_path: str,
    X_train: Sequence[Any],
    y_train: np.ndarray,
    X_test: Sequence[Any],
    timeout: int = DEFAULT_TIMEOUT_S,
) -> np.ndarray:
    scratch_id = uuid.uuid4().hex[:12]
    remote_dir = f"{REMOTE_ROOT}/{scratch_id}"

    with tempfile.TemporaryDirectory(prefix="ps3evolve-gpu-") as local_dir:
        local_dir_p = Path(local_dir)
        # saad304's upload stages files in a container-side tmp path keyed
        # ONLY by basename (not the full unique local path), so multiple
        # tracks running in parallel and both naming their local file
        # "candidate.py" collide and corrupt each other's upload. Every
        # filename here embeds scratch_id to guarantee a unique basename.
        candidate_name = f"candidate_{scratch_id}.py"
        input_name = f"input_{scratch_id}.pkl"
        candidate_local = local_dir_p / candidate_name
        candidate_local.write_text(Path(program_path).read_text(encoding="utf-8"), encoding="utf-8")
        payload = {
            # relative to the SLURM job's cwd (already cd'd into timesfm-classifier/),
            # NOT relative to the SSH login home dir like remote_dir is.
            "program_path": f"ps3evolve_gpu_scratch/{scratch_id}/{candidate_name}",
            "X_train": list(X_train),
            "y_train": np.asarray(y_train),
            "X_test": list(X_test),
        }
        input_local = local_dir_p / input_name
        with open(input_local, "wb") as f:
            pickle.dump(payload, f)

        mk = _remote(f"mkdir -p {remote_dir}")
        if mk.returncode != 0:
            raise CloudCandidateError(f"failed to create remote scratch dir: {mk.stderr[-300:]}")
        _remote_upload(str(candidate_local), f"{remote_dir}/{candidate_name}")
        _remote_upload(str(input_local), f"{remote_dir}/{input_name}")

        result_name = f"result_{scratch_id}.pkl"
        submit = _remote(
            f"cd timesfm-classifier && sbatch --parsable "
            f"--output={remote_dir}/slurm.out "
            f"--export=ALL,SCRATCH_ID={scratch_id},INPUT_NAME={input_name},RESULT_NAME={result_name} "
            f"jobs/ps3evolve_gpu.slurm"
        )
        if submit.returncode != 0 or not submit.stdout.strip():
            raise CloudCandidateError(f"sbatch submission failed: {submit.stderr[-500:]}")
        job_id = submit.stdout.strip().splitlines()[-1].split(";")[0].strip()

        deadline = time.time() + timeout
        consecutive_query_failures = 0
        while time.time() < deadline:
            q = _remote(f"squeue -j {job_id} -h -o %T", timeout=30)
            if q.returncode != 0:
                # transient SSH/squeue hiccup, NOT "job finished" -- retry,
                # but bail out if it keeps failing rather than looping forever
                consecutive_query_failures += 1
                if consecutive_query_failures >= 6:
                    raise CloudCandidateError(
                        f"squeue query failed {consecutive_query_failures}x in a row: {q.stderr[-300:]}"
                    )
                time.sleep(POLL_INTERVAL_S)
                continue
            consecutive_query_failures = 0
            state = q.stdout.strip()
            if not state:
                break  # query succeeded AND job is no longer queued/running -> finished
            time.sleep(POLL_INTERVAL_S)
        else:
            _remote_best_effort(f"scancel {job_id}", timeout=15)
            raise CloudCandidateError(f"job {job_id} timed out after {timeout}s and was cancelled")

        result_local = local_dir_p / result_name
        got = _remote_download(f"{remote_dir}/{result_name}", str(result_local))
        if not got:
            try:
                log = _remote(f"tail -c 2000 {remote_dir}/slurm.out 2>/dev/null || true", timeout=20)
                log_text = log.stdout[-1500:]
            except subprocess.TimeoutExpired:
                log_text = "(log tail fetch also timed out)"
            _remote_best_effort(f"rm -rf {remote_dir}", timeout=20)
            raise CloudCandidateError(
                f"job {job_id} produced no result (cluster log tail): {log_text}"
            )
        with open(result_local, "rb") as f:
            result = pickle.load(f)

        _remote_best_effort(f"rm -rf {remote_dir}", timeout=20)

        if result.get("error"):
            raise CloudCandidateError(result["error"])
        return np.asarray(result["predictions"])


def run_cv_cloud(X, y, splits, program_path, metric_fn, timeout: int = DEFAULT_TIMEOUT_S) -> dict:
    scores = []
    for tr_idx, te_idx in splits:
        X_tr = [X[i] for i in tr_idx]
        X_te = [X[i] for i in te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        preds = run_candidate_cloud(program_path, X_tr, y_tr, X_te, timeout=timeout)
        scores.append(float(metric_fn(y_te, preds)))
    return {"fold_scores": scores, "mean": float(np.mean(scores)) if scores else 0.0}


def run_cv_grouped_cloud(X, y, groups, splits, program_path, metric_fn, timeout: int = DEFAULT_TIMEOUT_S) -> dict:
    scores = []
    for tr_idx, te_idx in splits:
        X_tr = [X[i] for i in tr_idx]
        X_te = [X[i] for i in te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        g_te = groups[te_idx]
        preds = run_candidate_cloud(program_path, X_tr, y_tr, X_te, timeout=timeout)
        scores.append(float(metric_fn(y_te, preds, g_te)))
    return {"fold_scores": scores, "mean": float(np.mean(scores)) if scores else 0.0}


def code_length(program_path: str) -> int:
    return len(Path(program_path).read_text(encoding="utf-8"))
