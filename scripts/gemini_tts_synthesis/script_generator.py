#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script generator for Japanese paralinguistic breathy voice synthesis.

Employs professional Japanese film/TV drama screenwriter persona to generate
emotionally nuanced conversational scripts across Carlos T. Ishi's ATR taxonomy.
Communicates strictly with the local LLM endpoint (Fail-Fast, no external fallback),
and persists generated scripts via atomic sharded caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("script_generator")

# Local LLM service configuration (Strict: No fallback)
LOCAL_LLM_ENDPOINT = "http://127.0.0.1:8045"
LOCAL_LLM_KEY = "ENV_LOCAL_LLM_KEY_PLACEHOLDER"
LOCAL_LLM_MODEL = "gemini-3.8-flash-high"


def load_contexts(config_path: Path) -> List[Dict[str, Any]]:
    """Load paralinguistic context categories from YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("contexts", [])


def clean_transcript_for_tifa(text: str) -> str:
    """Remove inline angle tags like <breath>, <sigh>, <laugh> for phonetic alignment."""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def call_local_llm_json(
    prompt: str,
    endpoint: str = LOCAL_LLM_ENDPOINT,
    api_key: str = LOCAL_LLM_KEY,
    model: str = LOCAL_LLM_MODEL,
    timeout: int = 45,
) -> Dict[str, Any]:
    """Call strictly local LLM endpoint and return parsed JSON object.

    Raises RuntimeError immediately on connection or parsing failure (Fail-Fast).
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "あなたは日本の映画・連続テレビドラマを手掛ける一流のプロ脚本家・劇作家です。\n"
                    "日常の何気ない会話に潜む人間の機微、息づかい、言不由衷な本音や感情の揺らぎを、"
                    "リアルで生々しい生活感のあるセリフとして描き出すことを専門としています。\n\n"
                    "教科書的な不自然な日本語や硬い表現は一切排除し、登場人物の感情が高まり、"
                    "思わず声帯を震わせながら息が漏れ出す（気声・Breathy Voice）ようなドラマチックかつ自然な会話を構築してください。\n"
                    "必ず有効なJSONのみを出力し、Markdownのバッククォートや説明文は一切含めないでください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.45,
    }

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_data = resp.read().decode("utf-8")
            res = json.loads(raw_data)
            content = res["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        logger.error(
            f"\n{'='*70}\n"
            f"[FATAL] 本地文本 LLM 端口无法连接！({endpoint})\n"
            f"原因: {e}\n"
            f"请检查本地 LLM 服务（8045端口）是否已成功启动。\n"
            f"根据系统设计原则，禁止自动回退到公网接口，程序将直接中止退出。\n"
            f"{'='*70}\n"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"[FATAL] 本地 LLM 请求发生未知异常: {e}")
        sys.exit(1)

    # Clean markdown fences if any
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to parse LLM JSON output:\n{content}\nError: {e}")
        raise RuntimeError(f"Invalid JSON from local LLM: {e}") from e


def generate_single_script(
    context_info: Dict[str, Any],
    scenario_idx: int,
    endpoint: str = LOCAL_LLM_ENDPOINT,
    api_key: str = LOCAL_LLM_KEY,
    model: str = LOCAL_LLM_MODEL,
) -> Dict[str, Any]:
    """Generate a single structured dialogue script for a given paralinguistic sub-category."""
    scenarios = context_info.get("seed_scenarios", [])
    scenario = scenarios[scenario_idx % len(scenarios)] if scenarios else {"prompt_hint": "日常の自然な会話"}
    exemplars = context_info.get("few_shot_exemplars", [])

    few_shot_str = "\n".join([f"  - 例: 「{ex}」" for ex in exemplars]) if exemplars else "  - (なし)"
    tag = context_info.get("tag_recommendation", "<breath>")

    prompt = f"""日本の映画・ドラマの1シーンとして、自然な日本語会話の台詞を1件執筆してください。

【対象の副言語的気声（Breathy Voice）コンテキスト】
- 大分類: {context_info.get('macro_name', context_info['id'])}
- 子分類: {context_info.get('sub_name', context_info['id'])}
- 背景説明: {context_info['description']}
- 情景・シチュエーション: {scenario['prompt_hint']}
- ATR実測コーパス典範例（この呼吸感・自然さを参考にしてください）:
{few_shot_str}

【脚本執筆要件】
1. 人物Aの短い呼びかけ・導入発話（context_lead_in）と、人物Bの応答発話を作成してください。
2. 音声合成対象は【人物Bの発話のみ】です。
3. 人物Bの発話は、会話のテンポとして全体で3秒〜7秒程度（平仮名換算で20字〜45字程度）にしてください。
4. 人物Bの発話は必ず以下の2つのパートに論理的に分割してください：
   - modal_text: 導入部の通常の常態音（Modal Voice）で話す部分（例: 相槌、前置きなど、全体の30%〜50%）
   - breathy_text: 感情や意図が強く込められ、気声（Breathy Voice）で話す核心部分（全体の50%〜70%）
     ※重要：気声（Breathy Voice）は声帯の振動（音程・ピッチ）を維持したまま息が漏れる発声です。ささやき声（Whisper）ではありません。styleにwhisperやwhisper-likeという単語は絶対に含めないでください。
5. breathy_textの先頭には、息の漏れを表すタグ {tag} を挿入してください。
6. 出力は以下のJSONフォーマットのみを返してください：

{{
  "context_lead_in": "人物Aの台詞",
  "modal_text": "人物Bの前半の台詞（通常の平仮名/漢字、タグなし）",
  "modal_style": "{context_info['recommended_styles']['modal_part']}",
  "breathy_text_clean": "人物Bの後半の台詞（タグなしの純粋なテキスト）",
  "breathy_text_with_tag": "{tag} 人物Bの後半の台詞",
  "breathy_style": "{context_info['recommended_styles']['breathy_part']}"
}}
"""
    raw_res = call_local_llm_json(prompt, endpoint=endpoint, api_key=api_key, model=model)

    modal_text = raw_res["modal_text"].strip()
    breathy_clean = raw_res["breathy_text_clean"].strip()
    breathy_tagged = raw_res.get("breathy_text_with_tag", f"{tag} {breathy_clean}").strip()
    clean_full = clean_transcript_for_tifa(f"{modal_text} {breathy_clean}")

    script_id = f"script_{context_info['id']}_{scenario_idx+1:02d}"

    return {
        "script_id": script_id,
        "macro_id": context_info.get("macro_id", context_info["id"]),
        "macro_name": context_info.get("macro_name", ""),
        "sub_id": context_info.get("sub_id", context_info["id"]),
        "sub_name": context_info.get("sub_name", ""),
        "scenario_hint": scenario["prompt_hint"],
        "context_lead_in": raw_res.get("context_lead_in", ""),
        "modal_text": modal_text,
        "modal_style": raw_res.get("modal_style", context_info["recommended_styles"]["modal_part"]),
        "breathy_text_clean": breathy_clean,
        "breathy_text_with_tag": breathy_tagged,
        "breathy_style": raw_res.get("breathy_style", context_info["recommended_styles"]["breathy_part"]),
        "clean_full_text": clean_full,
    }


def save_atomic_script_shard(script_data: Dict[str, Any], shard_path: Path) -> None:
    """Atomically save single script dict to JSON."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=shard_path.parent, delete=False, encoding="utf-8") as tf:
        json.dump(script_data, tf, ensure_ascii=False, indent=2)
        temp_name = tf.name
    os.replace(temp_name, shard_path)


def generate_script_pool(
    config_path: Path,
    cache_dir: Path,
    scenarios_per_context: int = 3,
    endpoint: str = LOCAL_LLM_ENDPOINT,
    api_key: str = LOCAL_LLM_KEY,
    model: str = LOCAL_LLM_MODEL,
) -> List[Dict[str, Any]]:
    """Build or load atomic sharded script pool across all 17 ATR sub-categories.

    Args:
        config_path: Path to ishi_contexts.yaml
        cache_dir: Directory where script_{id}.json files are stored
        scenarios_per_context: Number of scenarios per sub-category (default 3 -> ~51 scripts total)

    Returns:
        List of script dictionary items.
    """
    contexts = load_contexts(config_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scripts: List[Dict[str, Any]] = []
    total_expected = len(contexts) * scenarios_per_context
    logger.info(f"Targeting script pool: {len(contexts)} sub-categories x {scenarios_per_context} = {total_expected} scripts.")

    for ctx in contexts:
        ctx_id = ctx["id"]
        for s_idx in range(scenarios_per_context):
            script_id = f"script_{ctx_id}_{s_idx+1:02d}"
            shard_file = cache_dir / f"{script_id}.json"

            if shard_file.exists():
                try:
                    with open(shard_file, "r", encoding="utf-8") as f:
                        cached_item = json.load(f)
                    scripts.append(cached_item)
                    continue
                except Exception as e:
                    logger.warning(f"Corrupted cache file {shard_file.name}, regenerating: {e}")

            logger.info(f"Generating new script [{len(scripts)+1}/{total_expected}] for {ctx_id} (#{s_idx+1})...")
            item = generate_single_script(
                context_info=ctx,
                scenario_idx=s_idx,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
            )
            save_atomic_script_shard(item, shard_file)
            scripts.append(item)

    logger.info(f"Script pool ready: {len(scripts)} scripts available in {cache_dir}.")
    return scripts


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import argparse

    parser = argparse.ArgumentParser(description="Standalone dialogue script pool generator (Screenwriter Persona).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "ishi_contexts.yaml",
        help="Path to ishi_contexts.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"E:\Corpus\Breathy\BreathyTTS\script_pool_cache"),
        help="Directory where script sharded JSON files are saved.",
    )
    parser.add_argument(
        "--count-per-sub",
        type=int,
        default=3,
        help="Number of scenarios to generate per sub-category (17 sub-categories * count).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=LOCAL_LLM_ENDPOINT,
        help="Local LLM endpoint URL (Fail-Fast, no fallback).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing script cache before generating.",
    )
    args = parser.parse_args()

    if args.clean and args.cache_dir.exists():
        logger.info(f"Cleaning existing script cache in {args.cache_dir}...")
        for old_f in args.cache_dir.glob("*.json"):
            old_f.unlink()

    pool = generate_script_pool(
        config_path=args.config,
        cache_dir=args.cache_dir,
        scenarios_per_context=args.count_per_sub,
        endpoint=args.endpoint,
    )
    print("\n" + "=" * 60)
    print(f"SCRIPT GENERATION COMPLETE!")
    print(f"- Total Scripts:   {len(pool)}")
    print(f"- Cache Directory: {args.cache_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
