# src/data_loader.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Constants / Types
# ----------------------------
EXAMS = ["Midterm 1", "Midterm 2", "Final"]
MODALITIES = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP", "tags"]


@dataclass(frozen=True)
class DatasetPaths:
    """
    Root points to the dataset folder:
      data/wearable-exam-stress/
        S1/
          Midterm 1/
            ACC.csv ...
        StudentGrades.txt
    """
    root: Path

    @property
    def grades_path(self) -> Path:
        return self.root / "StudentGrades.txt"

    def subject_dir(self, subject: str) -> Path:
        return self.root / normalize_subject(subject)

    def exam_dir(self, subject: str, exam: str) -> Path:
        if exam not in EXAMS:
            raise ValueError(f"exam must be one of {EXAMS}, got: {exam}")
        return self.subject_dir(subject) / exam


# ----------------------------
# Public API
# ----------------------------
def list_subjects(root: Path) -> List[str]:
    """Return subjects like ['S1','S2',...] found under root."""
    root = Path(root)
    subs = []
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"S\d+", p.name, flags=re.IGNORECASE):
            subs.append(p.name.upper())
    # sort by numeric id
    subs.sort(key=lambda s: int(re.sub(r"\D", "", s)))
    return subs


def list_exams(root: Path, subject: str) -> List[str]:
    """Return available exams for a subject (subset of EXAMS)."""
    root = Path(root)
    subj_dir = root / normalize_subject(subject)
    if not subj_dir.exists():
        return []
    exams = [e for e in EXAMS if (subj_dir / e).exists()]
    return exams


def load_exam_minimal(
    root: Path,
    subject: str,
    exam: str,
) -> Dict[str, pd.DataFrame]:
    """
    W1 loader: EDA/HR/ACC only.
    Returns:
      {'EDA': df(t,value), 'HR': df(t,value), 'ACC': df(t,x,y,z)}
    """
    paths = DatasetPaths(Path(root))
    exam_dir = paths.exam_dir(subject, exam)

    return {
        "EDA": read_signal_csv(exam_dir / "EDA.csv"),
        "HR":  read_signal_csv(exam_dir / "HR.csv"),
        "ACC": read_signal_csv(exam_dir / "ACC.csv"),
    }


def load_exam_modalities(
    root: Path,
    subject: str,
    exam: str,
    modalities: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    General loader: load any subset of modalities from an exam folder.

    modalities: e.g. ["EDA","HR","ACC","TEMP"]
    If None -> load all known modalities that exist.
    """
    paths = DatasetPaths(Path(root))
    exam_dir = paths.exam_dir(subject, exam)

    if modalities is None:
        modalities = MODALITIES

    out: Dict[str, pd.DataFrame] = {}
    for m in modalities:
        fname = f"{m}.csv"
        fpath = exam_dir / fname
        if fpath.exists():
            out[m] = read_signal_csv(fpath)
    return out


def compute_acc_mag(
    acc_df: pd.DataFrame,
    clip_percentile: Optional[float] = 99.9,
) -> pd.DataFrame:
    """
    Compute acceleration magnitude sqrt(x^2+y^2+z^2).
    clip_percentile:
      - None: no clipping
      - float (e.g. 99.9): clip extreme parsing artifacts/outliers for W1 plotting
    """
    need = {"t", "x", "y", "z"}
    if not need.issubset(acc_df.columns):
        raise ValueError(f"ACC df must contain {need}, got={set(acc_df.columns)}")

    mag = np.sqrt(
        acc_df["x"].to_numpy(dtype=float) ** 2
        + acc_df["y"].to_numpy(dtype=float) ** 2
        + acc_df["z"].to_numpy(dtype=float) ** 2
    )

    if clip_percentile is not None:
        upper = np.nanpercentile(mag, float(clip_percentile))
        mag = np.clip(mag, 0, upper)

    return pd.DataFrame({"t": acc_df["t"].to_numpy(dtype=float), "acc_mag": mag})


def load_grades(root: Path) -> pd.DataFrame:
    """
    Parse StudentGrades.txt (section format with possible non-UTF8 encoding).
    Returns dataframe:
      Subject | Midterm1 | Midterm2 | Final
      S1..S10 normalized.
    """
    paths = DatasetPaths(Path(root))
    grades_path = paths.grades_path
    if not grades_path.exists():
        raise FileNotFoundError(f"StudentGrades.txt not found at: {grades_path}")

    raw = grades_path.read_bytes()
    # cp1252 handles en-dash and common Windows chars well
    text = raw.decode("cp1252", errors="ignore")

    current: Optional[str] = None  # "Midterm1"|"Midterm2"|"Final"
    scores: Dict[str, Dict[str, float]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        upper = line.upper()

        # section headers
        if "GRADES" in upper and "MIDTERM" in upper and "1" in upper:
            current = "Midterm1"
            continue
        if "GRADES" in upper and "MIDTERM" in upper and "2" in upper:
            current = "Midterm2"
            continue
        if "GRADES" in upper and "FINAL" in upper:
            current = "Final"
            continue

        # skip separators
        if set(line) <= {"-", "–", "—"}:
            continue
        if current is None:
            continue

        # score line: S01 – 78  (dash could be -, – or —)
        m = re.search(r"^(S\s*0*\d+)\s*[–—-]\s*(\d+(\.\d+)?)\s*$", line)
        if not m:
            continue

        subj_raw = m.group(1).replace(" ", "").upper()  # e.g., S01
        score = float(m.group(2))
        subj = normalize_subject(subj_raw)              # -> S1

        scores.setdefault(subj, {})[current] = score

    if not scores:
        raise ValueError(
            "Failed to parse StudentGrades.txt (no scores extracted). "
            "File format may differ from expected section layout."
        )

    rows = []
    for subj, d in scores.items():
        rows.append(
            (
                subj,
                d.get("Midterm1", float("nan")),
                d.get("Midterm2", float("nan")),
                d.get("Final", float("nan")),
            )
        )

    df = pd.DataFrame(rows, columns=["Subject", "Midterm1", "Midterm2", "Final"])
    df = df.sort_values("Subject", key=lambda s: s.str.extract(r"(\d+)").astype(int)[0])
    return df.reset_index(drop=True)


def get_grade(grades_df: pd.DataFrame, subject: str, exam: str) -> float:
    """
    exam: 'Midterm 1'|'Midterm 2'|'Final'
    subject can be 'S1' or 'S01' etc.
    """
    subj = normalize_subject(subject)
    row = grades_df[grades_df["Subject"].astype(str).str.upper().eq(subj)]
    if row.empty:
        raise ValueError(f"Subject {subj} not found in grades.")
    row = row.iloc[0]

    if exam == "Midterm 1":
        return float(row["Midterm1"])
    if exam == "Midterm 2":
        return float(row["Midterm2"])
    if exam == "Final":
        return float(row["Final"])

    raise ValueError(f"Unknown exam: {exam}")


# ----------------------------
# Low-level parsing utilities
# ----------------------------
def normalize_subject(subject: str) -> str:
    """Normalize 'S01'/'s 001' -> 'S1' to match folder names."""
    s = str(subject).strip().upper().replace(" ", "")
    if not s.startswith("S"):
        raise ValueError(f"Invalid subject id: {subject}")
    num = int(re.sub(r"\D", "", s))
    return f"S{num}"


def read_signal_csv(path: Path) -> pd.DataFrame:
    """
    Robust reader for signal CSVs in wearable-exam-stress.

    Supports:
    (A) Empatica header format:
        line0: start unix timestamp (single float)
        line1: sampling rate (single float)
        remaining: values(1 col) or ACC(3 cols)
    (B) Timestamp-per-row:
        [timestamp, value] or [timestamp, x, y, z]
    (C) ACC 3-col:
        [x, y, z] only, no timestamp (we create t using assumed_fs=32Hz)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Detect (A): first line is a single float token (no commas)
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    tokens = [t.strip() for t in first_line.split(",") if t.strip() != ""]
    is_format_a = False
    if len(tokens) == 1:
        try:
            float(tokens[0])
            is_format_a = True
        except Exception:
            is_format_a = False

    if is_format_a:
        return _read_empatica_header_format(path)

    # Otherwise parse as numeric CSV (B/C)
    df = pd.read_csv(path, header=None)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    if df.shape[0] == 0:
        raise ValueError(f"No numeric data found in: {path}")

    ncol = df.shape[1]

    def _is_timestamp_like(x: np.ndarray) -> bool:
        med = np.nanmedian(x)
        return np.isfinite(med) and med > 1e6  # unix sec ~1e9

    # (B1): [timestamp, value]
    if ncol == 2:
        c0 = df.iloc[:, 0].to_numpy(dtype=float)
        c1 = df.iloc[:, 1].to_numpy(dtype=float)
        t = c0 - c0[0] if _is_timestamp_like(c0) else np.arange(df.shape[0], dtype=float)
        return pd.DataFrame({"t": t, "value": c1})

    # (B2): [timestamp, x, y, z] (or more cols)
    if ncol >= 4:
        c0 = df.iloc[:, 0].to_numpy(dtype=float)
        x = df.iloc[:, 1].to_numpy(dtype=float)
        y = df.iloc[:, 2].to_numpy(dtype=float)
        z = df.iloc[:, 3].to_numpy(dtype=float)
        t = c0 - c0[0] if _is_timestamp_like(c0) else np.arange(df.shape[0], dtype=float)
        return pd.DataFrame({"t": t, "x": x, "y": y, "z": z})

    # (C): 3 columns; often ACC=[x,y,z] without timestamp
    if ncol == 3:
        c0 = df.iloc[:, 0].to_numpy(dtype=float)
        c1 = df.iloc[:, 1].to_numpy(dtype=float)
        c2 = df.iloc[:, 2].to_numpy(dtype=float)

        # If it looks like timestamp triple at first row, treat as x,y,z anyway
        # We'll build t using assumed fs and let downstream clipping remove the first-row artifact.
        assumed_fs = 32.0
        t = np.arange(df.shape[0], dtype=float) / assumed_fs
        return pd.DataFrame({"t": t, "x": c0, "y": c1, "z": c2})

    raise ValueError(
        f"Unsupported CSV shape for {path.name}: {df.shape}. "
        "Expected Empatica header format, [timestamp,value], [timestamp,x,y,z], or ACC [x,y,z]."
    )


def _read_empatica_header_format(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    # start_unix = float(lines[0].strip())  # not used here
    fs = float(lines[1].strip())

    data_rows = []
    for ln in lines[2:]:
        parts = [p.strip() for p in ln.split(",") if p.strip() != ""]
        try:
            data_rows.append([float(x) for x in parts])
        except ValueError:
            continue

    data = np.asarray(data_rows, dtype=float)
    if data.ndim == 1:
        data = data[:, None]

    t = np.arange(data.shape[0], dtype=float) / fs

    if data.shape[1] == 1:
        return pd.DataFrame({"t": t, "value": data[:, 0]})
    if data.shape[1] == 3:
        return pd.DataFrame({"t": t, "x": data[:, 0], "y": data[:, 1], "z": data[:, 2]})

    cols = [f"c{i}" for i in range(data.shape[1])]
    out = pd.DataFrame(data, columns=cols)
    out.insert(0, "t", t)
    return out