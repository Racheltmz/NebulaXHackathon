"""Runs INSIDE the network-disabled Docker sandbox. Loads pickled
(program_path, X_train, y_train, X_test), imports the candidate module,
calls fit() then predict(), and writes the result back out. Never touches
y_test -- it isn't given to this process at all.
"""
import importlib.util
import pickle
import sys
import traceback

IN_PATH = "/work/input.pkl"
OUT_PATH = "/output/result.pkl"


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
