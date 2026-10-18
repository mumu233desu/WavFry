import os
import pytest
from wavfry.data.path_resolver import resolve_corpus_path, get_active_corpus_root
import wavfry.data.path_resolver as pr


def test_resolve_corpus_path_with_env(monkeypatch):
    monkeypatch.setenv("WAVFRY_CORPUS_ROOT", "/data/Corpus")
    pr._CACHED_CORPUS_ROOT = None  # Reset cache

    # Windows forward slash
    p1 = "E:/Corpus/CS-Dialogue/data/short_wav/WAVE/01.flac"
    assert resolve_corpus_path(p1) == "/data/Corpus/CS-Dialogue/data/short_wav/WAVE/01.flac"

    # Windows backslash
    p2 = r"E:\Corpus\Creaky\MELI_processed\audio\02.flac"
    assert resolve_corpus_path(p2) == "/data/Corpus/Creaky/MELI_processed/audio/02.flac"

    # Linux /Corpus path
    p3 = "/Corpus/Noise/QUT-NOISE_processed/noise/cafe.wav"
    assert resolve_corpus_path(p3) == "/data/Corpus/Noise/QUT-NOISE_processed/noise/cafe.wav"

    # None and empty
    assert resolve_corpus_path(None) is None

    # Unrelated path untouched
    unrelated = "/tmp/local_cache.pt"
    assert resolve_corpus_path(unrelated) == "/tmp/local_cache.pt"


def test_resolve_corpus_path_default_windows(monkeypatch):
    monkeypatch.delenv("WAVFRY_CORPUS_ROOT", raising=False)
    pr._CACHED_CORPUS_ROOT = None

    # In our local Windows environment, E:/Corpus exists:
    if os.path.exists("E:/Corpus"):
        p = r"E:\Corpus\Breathy\YSJ\audio\01.wav"
        resolved = resolve_corpus_path(p)
        assert resolved.replace("\\", "/").startswith("E:/Corpus/Breathy/YSJ")
