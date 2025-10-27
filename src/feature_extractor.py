import numpy as np
import pandas as pd
import random, datetime, warnings, os
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from tqdm import tqdm

warnings.filterwarnings("ignore")
client = Client("IRIS")  # uses IRIS FDSN

def p_wave_features_calc(window: np.ndarray, dt: float) -> dict:
    if len(window) == 0:
        return {k: np.nan for k in [
            "pkev12","pkev23","durP","tauPd","tauPt","PDd","PVd","PAd","PDt","PVt","PAt",
            "ddt_PDd","ddt_PVd","ddt_PAd","ddt_PDt","ddt_PVt","ddt_PAt"
        ]}

    durP = len(window) * dt
    PDd = np.max(window) - np.min(window)
    PVd = np.max(np.abs(np.gradient(window) / dt))
    PAd = np.mean(np.abs(window))
    PDt = np.max(window)
    PVt = np.max(np.gradient(window) / dt)
    PAt = np.sqrt(np.mean(window ** 2))
    tauPd = durP / PDd if PDd != 0 else 0
    tauPt = durP / PDt if PDt != 0 else 0

    ddt = lambda x: np.mean(np.abs(np.gradient(x))) if len(x) > 1 else 0

    ddt_PDd = ddt(window)
    grad = np.gradient(window) / dt
    ddt_PVd = ddt(grad)
    ddt_PAd = ddt(np.abs(window))
    ddt_PDt = ddt(np.maximum(window, 0))
    ddt_PVt = ddt(grad)
    ddt_PAt = ddt(window ** 2)

    pkev12 = np.sum(window ** 2) / len(window)
    pkev23 = np.sum(np.abs(window)) / len(window)

    return {
        "pkev12": pkev12, "pkev23": pkev23,
        "durP": durP, "tauPd": tauPd, "tauPt": tauPt,
        "PDd": PDd, "PVd": PVd, "PAd": PAd,
        "PDt": PDt, "PVt": PVt, "PAt": PAt,
        "ddt_PDd": ddt_PDd, "ddt_PVd": ddt_PVd,
        "ddt_PAd": ddt_PAd, "ddt_PDt": ddt_PDt,
        "ddt_PVt": ddt_PVt, "ddt_PAt": ddt_PAt
    }

def extract_from_iris(num_samples=10, stations=None, network="IU", year_choices=[2022,2023,2024], out_csv=None):
    if stations is None:
        stations = ["ANMO", "COR", "MAJO", "KBL"]

    records = []
    for i in tqdm(range(num_samples), desc="Extracting features"):
        starttime = UTCDateTime(datetime.datetime(
            random.choice(year_choices),
            random.randint(1, 12),
            random.randint(1, 25),
            random.randint(0, 21), 0, 0
        ))
        endtime = starttime + 2 * 3600

        try:
            st = None
            for station in stations:
                try:
                    st = client.get_waveforms(network, station, "*", "BHZ", starttime, endtime)
                    if st and len(st) > 0:
                        break
                except Exception:
                    continue

            if st is None or len(st) == 0:
                continue

            tr = st[0]
            tr.detrend("demean")
            tr.filter("bandpass", freqmin=0.5, freqmax=20.0)
            dt = tr.stats.delta

            cft = classic_sta_lta(tr.data, int(1 / dt), int(10 / dt))
            trig = trigger_onset(cft, 2.5, 1.0)
            if len(trig) == 0:
                continue

            p_index = trig[0][0]
            win = int(2.0 / dt)
            p_window = tr.data[p_index:p_index + win]

            if len(p_window) < 10:
                continue

            feats = p_wave_features_calc(p_window, dt)
            feats.update({
                "station": tr.stats.station,
                "network": tr.stats.network,
                "starttime": str(starttime),
                "sampling_rate": tr.stats.sampling_rate
            })
            records.append(feats)

        except Exception:
            continue

    df = pd.DataFrame(records)
    if out_csv and not df.empty:
        df.to_csv(out_csv, index=False)
    return df

# CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract P-wave features from IRIS")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--out", type=str, default="data/p_wave_features_dataset.csv")
    args = parser.parse_args()
    df = extract_from_iris(num_samples=args.num_samples, out_csv=args.out)
    print(f"Extracted {len(df)} samples → {args.out}")
