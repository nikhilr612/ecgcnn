"""
Open MIT-BIH data with `wfdb`, create windowed signals for all annotated heartbeats.
"""

import wfdb
import numpy as np
import polars as pl
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

half_window = 64
patients = [str(i) for i in range(100, 125)] + [str(i) for i in range(200, 235)]
tuples = []

aami5_char_to_ordinal = {
    "N": 0,  # Normal
    "R": 0,  # Right bundle branch block beat -> Normal
    "L": 0,  # Left bundle branch block beat -> Normal
    "e": 0,  # Atrial escape beat -> Normal
    "j": 0,  # Nodal escape beat -> Normal
    "V": 1,  # Premature ventricular contraction
    "E": 1,  # Ventricular Escape beat -> V
    "I": 1,  # R-on-T Premature Ventricular Contraction -> V
    "a": 2,  # Aberrated atrial premature beat -> S
    "n": 2,  # Nodal Premature beat -> S
    "A": 2,  # Atrial premature beat -> S
    "S": 2,  # Supraventricular premature or ectopic beat -> S
    "Q": 4,  # Unclassifiable -> Q
    "/": 3,  # Paced beat -> F
    "F": 3,  # Fusion of Ventricular and normal beat -> F
    "f": 3,  # Fusion of paced and normal beat -> F
    "[": 5,  # Start of ventricular flutter -> VF
    "!": 5,  # Ventricular flutter wave -> VF
    "]": 5,  # End of ventricular flutter -> VF
}


# TODO: See if you can extract more features like sex, and age from annotations' comments.
@dataclass
class EcgItem:
    signal_a: list[float]
    signal_b: list[float]
    shift_a: float
    scale_a: float
    shift_b: float
    scale_b: float
    signame_a: str
    signame_b: str
    annotation: int
    aami5: int


EPS = 1e-7


def process_signal(signals, sample_index):
    window = signals[max(sample_index - half_window, 0) : sample_index + half_window]
    shifted = window - (shift := window.mean())
    left_pad = half_window - (len(window) // 2)
    right_pad = 2 * half_window - len(window) - left_pad
    scale = window.std()
    if scale < EPS:
        scale = 1
    padded = np.pad(shifted, (left_pad, right_pad), mode="constant", constant_values=0)
    assert padded is not None, "Padding somehow made the signal null"
    scaled = padded * 1 / scale
    assert scaled is not None, "Signal somehow became null"
    return (scaled, shift, scale)


def process_patient(record_name: str, tuples: list[EcgItem]) -> list[EcgItem]:
    record = wfdb.rdrecord("./data/raw/" + record_name)
    annos = wfdb.rdann("./data/raw/" + record_name, "atr")
    signals_a = record.p_signal[:, 0]  # type: ignore
    signals_b = record.p_signal[:, 1]  # type: ignore
    name_a, name_b = record.sig_name  # type: ignore
    res = []
    for symbol, sample_index in zip(annos.symbol, annos.sample):  # type: ignore
        padded_a, shift_a, scale_a = process_signal(signals_a, sample_index)
        padded_b, shift_b, scale_b = process_signal(signals_b, sample_index)
        item = EcgItem(
            signal_a=list(padded_a),
            shift_a=shift_a,
            scale_a=scale_a,
            signal_b=list(padded_b),
            shift_b=shift_b,
            scale_b=scale_b,
            annotation=ord(symbol[0]),
            aami5=aami5_char_to_ordinal.get(symbol, 4),
            signame_a=name_a,
            signame_b=name_b,
        )
        res.append(item)

    return res


def safe_process_patient(patient):
    try:
        return process_patient(patient, [])
    except Exception as e:
        print(f"Failed to process patient {patient}: {e}")
        return []


with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(safe_process_patient, patient): patient for patient in patients
    }

    for future in as_completed(futures):
        patient = futures[future]
        try:
            result = future.result()
            tuples.extend(result)
        except Exception as e:
            print(f"Error processing patient {patient}: {e}")

df = pl.DataFrame(
    (asdict(item) for item in tuples),
    schema={
        "signal_a": pl.Array(pl.Float64, 2 * half_window),
        "signal_b": pl.Array(pl.Float64, 2 * half_window),
        "shift_a": pl.Float64,
        "scale_a": pl.Float64,
        "shift_b": pl.Float64,
        "scale_b": pl.Float64,
        "signame_a": pl.Utf8,
        "signame_b": pl.Utf8,
        "annotation": pl.UInt8,
        "aami5": pl.UInt8,
    },
    orient="row",
)

df.write_parquet("data/mitbih.parquet")
