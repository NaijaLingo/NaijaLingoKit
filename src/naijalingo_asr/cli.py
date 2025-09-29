import argparse
from .transcriber import transcribe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NaijaLingo ASR CLI")
    p.add_argument("--audio_path", required=True, help="Path to audio file")
    p.add_argument("--language", required=True, help="Language code (yo, ig, ha, en)")
    p.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    p.add_argument("--compute-type", dest="compute_type", default="auto", help="auto|float16|int8|int8_float16")
    p.add_argument("--beam-size", dest="beam_size", type=int, default=5)
    p.add_argument("--no-vad", dest="vad_filter", action="store_false")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--initial-prompt")
    p.add_argument("--log-level", default=None, help="Set log level (DEBUG, INFO, WARNING, ERROR)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    text = transcribe(
        args.audio_path,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        temperature=args.temperature,
        initial_prompt=args.initial_prompt,
        log_level=args.log_level,
    )
    print(text)


if __name__ == "__main__":
    main()
