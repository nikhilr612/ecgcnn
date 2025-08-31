"""
Open MIT-BIH data with `wfdb`, create windowed signals for all annotated heartbeats.
"""

import wfdb
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed

half_window = 64
patients = [str(i) for i in range(100, 125)] + [str(i) for i in range(200, 235)]
tuples = []


def process_patient(
    record_name: str,
    tuples: list,
    lead: int = 0,
) -> list[tuple[list[float], float, float, str]]:
    record = wfdb.rdrecord("./data/" + record_name)
    annos = wfdb.rdann("./data/" + record_name, "atr")
    signals = record.p_signal[:, lead]  # type: ignore
    res = []
    for symbol, sample_index in zip(annos.symbol, annos.sample):  # type: ignore
        window = signals[
            max(sample_index - half_window, 0) : sample_index + half_window
        ]
        shifted = window - (shift := window.mean())
        left_pad = half_window - (len(window) // 2)
        right_pad = 2 * half_window - len(window) - left_pad
        padded = (
            np.pad(shifted, (left_pad, right_pad), mode="constant", constant_values=0)
            * 1
            / (scale := window.std())
        )
        res.append((list(padded), shift, scale, symbol))

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

ordinalizer_dict = {}
ordinal = 0
for i, (signal, shift, scale, symbol) in enumerate(tuples):
    if symbol not in ordinalizer_dict:
        ordinalizer_dict[symbol] = ordinal
        ordinal += 1
    tuples[i] = (signal, shift, scale, ordinalizer_dict[symbol])

df = pl.DataFrame(
    tuples,
    schema={
        "signal": pl.List(pl.Float64),
        "shift": pl.Float64,
        "scale": pl.Float64,
        "classification": pl.Int64,
    },
    orient="row",
)

del tuples
print(f"Ordinal mapping {ordinalizer_dict}")

df.write_parquet("mitbih.parquet")
