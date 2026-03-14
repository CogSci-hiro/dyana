from __future__ import annotations

import argparse
import sys

from dyana.io.audio import detect_channel_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect whether a stereo WAV contains the same signal on both channels.")
    parser.add_argument("wav_path", help="Path to a stereo WAV file.")
    parser.add_argument("--threshold", type=float, default=0.98, help="Correlation threshold for classifying channels as the same signal.")
    parser.add_argument("--max-samples", type=int, default=1_000_000, help="Maximum number of samples used for correlation.")
    args = parser.parse_args()

    try:
        label, correlation = detect_channel_similarity(
            args.wav_path,
            threshold=args.threshold,
            max_samples=args.max_samples,
        )
    except Exception as exc:
        print(f"{args.wav_path} -> error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    signal_label = "same signal" if label == "same" else "different signals"
    print(f"{args.wav_path} -> correlation={correlation:.3f} -> {signal_label}")


if __name__ == "__main__":
    main()
