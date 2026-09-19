"""Runs ON THE CLUSTER (via SLURM, on a GPU compute node) inside the
candidate's own scratch directory. Loads pickled (program_path, X_train,
y_train, X_test), imports the candidate module, calls fit() then predict(),
and writes the result back out. Never touches y_test.

Unlike the local Docker sandbox, this process runs on a shared academic
cluster with normal outbound network access (no --network none equivalent
available here) -- see cloud_harness.py's docstring for the honest
anti-cheat disclosure for this track. The one guarantee that DOES still
hold unconditionally: true labels for whatever this call is being scored on
are never transmitted here in the first place, so there is nothing for a
network-capable candidate to "phone home" for that it doesn't already have.
"""
import importlib.util
import pickle
import sys
import traceback

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "input.pkl"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "result.pkl"


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    with open(IN_PATH, "rb") as f:
        payload = pickle.load(f)
    program_path = payload["program_path"]
    X_train, y_train, X_test = payload["X_train"], payload["y_train"], payload["X_test"]

    result = {"predictions": None, "error": None, "traceback": None}
    try:
        module = load_module(program_path)
        model = module.Model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result["predictions"] = preds
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    with open(OUT_PATH, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
