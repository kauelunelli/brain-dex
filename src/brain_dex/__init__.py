from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_default_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-dex",
        description="Predicao de resposta cerebral (fMRI) a partir de video usando TRIBE v2.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Caminho para o video de entrada (.mp4 recomendado).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Pasta para cache de pesos/modelos do Hugging Face.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps", "openvino"],
        default="auto",
        help="Dispositivo de inferencia. Em NVIDIA, use 'cuda'. OpenVINO nao e suportado pelo TRIBE v2 atual.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Pasta de saida para CSV e JSON de resumo.",
    )
    parser.add_argument(
        "--save-full-preds",
        action="store_true",
        help="Se definido, salva tambem a matriz completa de predicoes em .npy.",
    )
    return parser


def _summarize_predictions(preds: np.ndarray) -> dict[str, float | int]:
    flat = preds.reshape(-1)
    return {
        "timesteps": int(preds.shape[0]),
        "vertices": int(preds.shape[1]),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "mean_abs": float(np.mean(np.abs(flat))),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    video_path = args.video.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video nao encontrado: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from tribev2 import TribeModel

    device = _detect_default_device() if args.device == "auto" else args.device
    if device == "openvino":
        raise SystemExit(
            "OpenVINO nao e suportado pela implementacao atual do TRIBE v2. "
            "Use --device cpu ou --device cuda."
        )

    print(f"[1/4] Carregando modelo TRIBE v2 em {device}...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(cache_dir),
        device=device,
    )

    print("[2/4] Extraindo eventos do video...")
    events_df = model.get_events_dataframe(video_path=str(video_path))

    print("[3/4] Rodando inferencia...")
    preds, segments = model.predict(events=events_df)

    print("[4/4] Salvando resultados...")
    summary = _summarize_predictions(preds)

    summary_path = output_dir / "summary.json"
    events_path = output_dir / "events.csv"
    segments_path = output_dir / "segments.csv"

    events_df.to_csv(events_path, index=False)
    pd.DataFrame(segments).to_csv(segments_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_full_preds:
        np.save(output_dir / "predictions.npy", preds)

    print("Analise concluida.")
    print(f"- Summary: {summary_path}")
    print(f"- Events:  {events_path}")
    print(f"- Segments:{segments_path}")
    if args.save_full_preds:
        print(f"- Preds:   {output_dir / 'predictions.npy'}")
