"""Download diverse UCR/UEA classification datasets and emit a manifest."""
import argparse, json, os
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    from aeon.datasets import load_classification
    names = [
        "ECG200", "Ham", "InsectWingbeatSound", "BirdChicken", "CBF",
        "ElectricDevices", "FordA", "Wafer", "ArrowHead", "Coffee",
        "GunPoint", "ItalyPowerDemand", "Lightning2", "MedicalImages",
        "MoteStrain", "OliveOil", "OSULeaf", "TwoPatterns", "Yoga",
        "FaceAll", "Chinatown", "DiatomSizeReduction",
        "DistalPhalanxOutlineCorrect", "ECGFiveDays",
        "GunPointOldVersusYoung", "InsectEPGRegularTrain", "Meat",
        "MiddlePhalanxOutlineCorrect", "NonInvasiveFetalECGThorax1",
        "Plane", "ProximalPhalanxOutlineCorrect", "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2", "StarLightCurves", "Symbols",
        "ToeSegmentation1", "ToeSegmentation2", "Trace",
        "UWaveGestureLibraryAll", "WordSynonyms", "Worms",
        # Additional heterogeneous UCR tasks for the next repurposing cohort.
        "ACSF1", "Adiac", "Beef", "Car", "CricketX", "CricketY",
        "CricketZ", "Fungi", "Herring", "Lightning7",
        "MelbournePedestrian", "Phoneme", "SemgHandGenderCh2",
        "ShapeletSim", "SmallKitchenAppliances", "Strawberry",
        "SwedishLeaf", "TDD", "TwoLeadECG", "UnitTest",
    ]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); manifest = {}
    for name in names:
        try:
            tx, y = load_classification(name, split="train")
            vx, vy = load_classification(name, split="test")
            # Store one object array per split to preserve variable channel/length conventions.
            txp, typ = out / f"{name}_train_x.npy", out / f"{name}_train_y.npy"
            vxp, vyp = out / f"{name}_test_x.npy", out / f"{name}_test_y.npy"
            np.save(txp, tx.astype("float32")); np.save(typ, y); np.save(vxp, vx.astype("float32")); np.save(vyp, vy)
            manifest[name] = {"train_x": str(txp), "train_y": str(typ), "test_x": str(vxp), "test_y": str(vyp)}
            print(name, tx.shape, vx.shape, flush=True)
        except Exception as e:
            print("SKIP", name, repr(e), flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)

if __name__ == "__main__": main()
