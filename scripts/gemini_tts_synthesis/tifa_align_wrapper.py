#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper around YouTube-Collector's TIFA Japanese forced aligner.

Executes TIFA in directory mode using E:\Coding\YouTube-Collector\.venv to generate
TextGrid files ({sample_id}.TextGrid).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("tifa_align_wrapper")

TIFA_PYTHON_EXE = Path(r"E:\Coding\YouTube-Collector\.venv\Scripts\python.exe")
TIFA_REPO_DIR = Path(r"E:\Coding\YouTube-Collector\third_party\TIFA")
TIFA_MODEL_PT = Path(r"E:\Coding\YouTube-Collector\models\tifa\TIFA-1.0-ST\model.pt")
TIFA_ENTRY_PY = Path(r"E:\Coding\YouTube-Collector\src\tifa_infer_entry.py")


class TIFAAlignmentWrapper:
    """Invokes TIFA forced aligner across audio-text pairs."""

    def __init__(
        self,
        python_exe: Path = TIFA_PYTHON_EXE,
        repo_dir: Path = TIFA_REPO_DIR,
        model_pt: Path = TIFA_MODEL_PT,
        entry_py: Path = TIFA_ENTRY_PY,
        language: str = "ja",
    ):
        self.python_exe = python_exe
        self.repo_dir = repo_dir
        self.model_pt = model_pt
        self.entry_py = entry_py
        self.language = language

        if not self.python_exe.exists():
            raise FileNotFoundError(f"TIFA Python executable not found: {self.python_exe}")
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"TIFA repo directory not found: {self.repo_dir}")
        if not self.model_pt.exists():
            raise FileNotFoundError(f"TIFA model checkpoint not found: {self.model_pt}")
        if not self.entry_py.exists():
            raise FileNotFoundError(f"TIFA entry wrapper not found: {self.entry_py}")

    def align_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        batch_size: int = 4,
    ) -> Dict[str, Path]:
        """Run TIFA over input_dir ({sample_id}.wav + {sample_id}.txt).

        Returns:
            Dict mapping sample_id -> Path to generated .TextGrid file.
        """
        input_dir = input_dir.resolve()
        out_dir = (output_dir or input_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.flac"))
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {}

        logger.info(f"Running TIFA aligner on {len(audio_files)} files in {input_dir.name}...")

        cmd = [
            str(self.python_exe),
            str(self.entry_py),
            str(self.repo_dir),
            str(input_dir),
            "-m",
            str(self.model_pt),
            "-l",
            self.language,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            "0",
            "-o",
            str(out_dir),
        ]

        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )

        tail: List[str] = []
        for line in proc.stdout:
            clean_line = line.rstrip()
            tail.append(clean_line)
            if len(tail) > 25:
                tail.pop(0)

        proc.wait()
        if proc.returncode != 0:
            err_log = "\n".join(tail[-15:])
            raise RuntimeError(f"TIFA alignment failed (exit code {proc.returncode}):\n{err_log}")

        # Locate produced TextGrids
        textgrid_map: Dict[str, Path] = {}
        for wav in audio_files:
            tg_candidates = [
                out_dir / f"{wav.stem}.TextGrid",
                input_dir / f"{wav.stem}.TextGrid",
            ]
            for tg in tg_candidates:
                if tg.exists():
                    textgrid_map[wav.stem] = tg
                    break

        logger.info(f"TIFA alignment completed: {len(textgrid_map)}/{len(audio_files)} TextGrids produced.")
        return textgrid_map
