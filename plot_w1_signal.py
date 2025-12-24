import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import load_exam_minimal, compute_acc_mag, load_grades, get_grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data/wearable-exam-stress")
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--exam", default="Midterm 1", choices=["Midterm 1", "Midterm 2", "Final"])
    ap.add_argument("--out", default="figures/S1_midterm1_w1.png")
    args = ap.parse_args()

    root = Path(args.root)
    grades_df = load_grades(root)   # ✅ 傳 root 就好

    data = load_exam_minimal(root, args.subject, args.exam)
    grade = get_grade(grades_df, args.subject, args.exam)

    eda = data["EDA"]
    hr = data["HR"]
    acc_mag = compute_acc_mag(data["ACC"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(eda["t"] / 60.0, eda["value"])
    ax1.set_ylabel("EDA")

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(hr["t"] / 60.0, hr["value"])
    ax2.set_ylabel("HR")

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(acc_mag["t"] / 60.0, acc_mag["acc_mag"])
    ax3.set_ylabel("ACC |mag|")
    ax3.set_xlabel("Time since exam start (min)")

    plt.suptitle(f"{args.subject} – {args.exam} | Grade: {grade}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
