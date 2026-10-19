#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script generator for Japanese paralinguistic breathy voice synthesis.

Employs professional Japanese film/TV drama screenwriter persona to generate
emotionally nuanced conversational scripts across Carlos T. Ishi's ATR taxonomy.
Communicates strictly with the local LLM endpoint (Fail-Fast, no external fallback),
and persists generated scripts via atomic sharded caching.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("script_generator")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Local LLM service configuration (Strict: No fallback)
LOCAL_LLM_ENDPOINT = os.environ.get("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8045")
LOCAL_LLM_KEY = os.environ.get("LOCAL_LLM_KEY", "")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "gemini-3.8-flash-high")


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
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Call strictly local LLM endpoint and return parsed JSON object.

    Raises RuntimeError immediately on connection or parsing failure (Fail-Fast).
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
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
        "temperature": 0.5,
    }
    if seed is not None:
        body["seed"] = seed

    max_retries = 3
    content = ""
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_data = resp.read().decode("utf-8")
                res = json.loads(raw_data)
                content = res["choices"][0]["message"]["content"].strip()
                break
        except urllib.error.URLError as e:
            # If target machine actively refused connection, fail fast immediately
            err_str = str(e)
            if "10061" in err_str or "Connection refused" in err_str:
                logger.error(
                    f"\n{'='*70}\n"
                    f"[FATAL] 本地文本 LLM 端口无法连接！({endpoint})\n"
                    f"原因: {e}\n"
                    f"请检查本地 LLM 服务（8045端口）是否已成功启动。\n"
                    f"根据系统设计原则，禁止自动回退到公网接口，程序将直接中止退出。\n"
                    f"{'='*70}\n"
                )
                os._exit(1)
            elif attempt < max_retries:
                logger.warning(f"Transient error calling local LLM (attempt {attempt}/{max_retries}): {e}. Retrying in 2s...")
                time.sleep(2.0)
            else:
                logger.error(f"Local LLM request failed after {max_retries} attempts: {e}")
                raise RuntimeError(f"Local LLM request failed after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Unexpected error (attempt {attempt}/{max_retries}): {e}. Retrying in 2s...")
                time.sleep(2.0)
            else:
                logger.error(f"Local LLM request encountered unexpected error: {e}")
                raise RuntimeError(f"Local LLM error: {e}")

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


def validate_script_data(raw_res: Any) -> Tuple[bool, str]:
    """Validate JSON schema and content requirements for dialogue script."""
    if not isinstance(raw_res, dict):
        return False, "Output is not a JSON object"
    for field in ["context_lead_in", "modal_text", "breathy_text_clean"]:
        if field not in raw_res or not isinstance(raw_res[field], str):
            return False, f"Missing or non-string field: '{field}'"
        if len(raw_res[field].strip()) == 0:
            return False, f"Field '{field}' is empty"

    total_len = len(raw_res["modal_text"].strip()) + len(raw_res["breathy_text_clean"].strip())
    if total_len < 25:
        return False, f"Script text too short ({total_len} chars), expected >= 25 chars for ~5s+ dialogue"

    return True, ""


def generate_single_script(
    context_info: Dict[str, Any],
    scenario_idx: int,
    endpoint: str = LOCAL_LLM_ENDPOINT,
    api_key: str = LOCAL_LLM_KEY,
    model: str = LOCAL_LLM_MODEL,
    max_attempts: int = 4,
) -> Dict[str, Any]:
    """Generate a single structured dialogue script for a given paralinguistic sub-category with retry validation."""
    scenarios = context_info.get("seed_scenarios", [])
    scenario = scenarios[scenario_idx % len(scenarios)] if scenarios else {"prompt_hint": "日常の自然な会話"}
    exemplars = context_info.get("few_shot_exemplars", [])
    few_shot_str = "\n".join([f"  - 例: 「{ex}」" for ex in exemplars]) if exemplars else "  - (なし)"
    tag = context_info.get("tag_recommendation", "<breath>")

    prompt = f"""日本の映画・ドラマの1シーンとして、自然な日本語会話の台詞を1件執筆してください。

【対象の副言語的気声（Breathy Voice）コンテキスト】
- 大分類: {context_info.get('macro_name', context_info['id'])}
- 子分類: {context_info.get('sub_name', context_info['id'])}
- 感情・発話意図の背景: {context_info['description']}
- 台詞の参考例（実際の会話の空気感・発声のお手本）:
{few_shot_str}

【話題（トピック）の自由選定と独立性に関する重要指示】
1. 上記の「台詞の参考例」は、あくまで声のトーンや息づかい、口語表現の自然さ（リアリティ）を示すためのお手本です。
   ※最重要：参考例に含まれる具体的な話題や単語（例: カロリー、特定の食べ物や名詞など）は決して真似・流用しないでください。
2. 今回の対話（通番 #{scenario_idx+1}）で取り上げる【具体的な話題・出来事・シチュエーション】は、プロの脚本家としてあなた自身が自由にゼロから考案・選定してください。
   （例: 職場での想定外のミスや成功、趣味のこだわり、家族やペットの珍しい行動、旅行先でのハプニング、日用品の買い物、秘密の告白、天気の急変など、日常のあらゆる多彩なテーマから自由に選んでください）
3. 同じ子分類であっても、通番ごとに毎回まったく異なる人間関係（同僚、夫婦、親子、幼馴染、初対面の人など）と新鮮な話題を設定してください。

【脚本執筆要件】
1. 人物Aの短い呼びかけ・導入発話（context_lead_in）と、人物Bの応答発話を作成してください。
2. 音声合成対象は【人物Bの発話のみ】です。
3. 人物Bの発話は、短すぎず感情と状況を豊かに表現するため、全体で5秒〜15秒程度（最大20秒以内、平仮名換算で35字〜110字程度）にしてください。
4. 人物Bの発話は必ず以下の2つのパートに論理的に分割してください：
   - modal_text: 導入部の通常の常態音（Modal Voice）で話す部分（例: 相槌、前置き、状況の受け止めなど、全体の30%〜40%）
   - breathy_text: 感情や意図が強く込められ、息が漏れ出る気声（Breathy Voice）で話す核心部分（全体の60%〜70%）
     ※重要：気声（Breathy Voice）は声帯の振動（音程・ピッチ）を維持したまま息が漏れる発声です。ささやき声（Whisper）ではありません。styleにwhisperやwhisper-likeという単语は絶対に含めないでください。
5. breathy_textの先頭には、息の漏れを表すタグ {tag} を挿入してください。
6. 出力は以下のJSONフォーマットのみを返してください：

{{
  "scenario_topic": "あなたが考案した独自の話題・シチュエーション（簡潔に1行）",
  "context_lead_in": "人物Aの台詞",
  "modal_text": "人物Bの前半の台詞（通常の平仮名/漢字、タグなし）",
  "modal_style": "{context_info['recommended_styles']['modal_part']}",
  "breathy_text_clean": "人物Bの後半の台詞（タグなしの純粋なテキスト）",
  "breathy_text_with_tag": "{tag} 人物Bの後半の台詞",
  "breathy_style": "{context_info['recommended_styles']['breathy_part']}"
}}
"""
    raw_res = None
    used_seed = None
    for attempt in range(1, max_attempts + 1):
        seed_val = random.randint(100000, 999999)
        prompt_with_seed = f"{prompt.rstrip()}\n\n[seed: {seed_val}]\n"
        try:
            res = call_local_llm_json(prompt_with_seed, endpoint=endpoint, api_key=api_key, model=model, seed=seed_val)
            valid, reason = validate_script_data(res)
            if not valid:
                logger.warning(
                    f"Script validation failed for {context_info['id']} (#{scenario_idx+1}, attempt {attempt}/{max_attempts}): {reason}. Retrying..."
                )
                time.sleep(1.0)
                continue
            raw_res = res
            used_seed = seed_val
            break
        except Exception as e:
            logger.warning(
                f"Error calling local LLM for {context_info['id']} (#{scenario_idx+1}, attempt {attempt}/{max_attempts}): {e}. Retrying..."
            )
            time.sleep(1.0)

    if raw_res is None:
        raise RuntimeError(
            f"Failed to generate valid script for {context_info['id']} (#{scenario_idx+1}) after {max_attempts} attempts."
        )

    modal_text = raw_res["modal_text"].strip()
    breathy_clean = raw_res["breathy_text_clean"].strip()

    # Sanitize whisper mentions in style if any (preserving negative constraints)
    breathy_style = raw_res.get("breathy_style", context_info["recommended_styles"]["breathy_part"]).strip()
    breathy_style = re.sub(r"(?<!not\s)(?<!never\s)\bwhisper(?:-like|ing)?\b", "voiced breathy", breathy_style, flags=re.IGNORECASE)
    breathy_style = breathy_style.replace("strictly NOT voiced breathy", "strictly NOT unvoiced whisper")

    breathy_tagged = raw_res.get("breathy_text_with_tag", "").strip()
    if not breathy_tagged.startswith(tag):
        breathy_tagged = f"{tag} {breathy_clean}"

    clean_full = clean_transcript_for_tifa(f"{modal_text} {breathy_clean}")
    # Assert strict character consistency (ignoring whitespace and tags)
    if re.sub(r"\s+", "", modal_text + breathy_clean) != re.sub(r"\s+", "", clean_full):
        raise ValueError(f"Consistency assertion failed: clean_full != modal_text + breathy_clean for {context_info['id']}")
    script_id = f"script_{context_info['id']}_{scenario_idx+1:02d}"

    topic_desc = raw_res.get("scenario_topic", "").strip() or scenario.get("prompt_hint", f"独自創作シナリオ #{scenario_idx+1}")

    return {
        "script_id": script_id,
        "seed": used_seed,
        "macro_id": context_info.get("macro_id", context_info["id"]),
        "macro_name": context_info.get("macro_name", ""),
        "sub_id": context_info.get("sub_id", context_info["id"]),
        "sub_name": context_info.get("sub_name", ""),
        "scenario_hint": topic_desc,
        "context_lead_in": raw_res.get("context_lead_in", "").strip(),
        "modal_text": modal_text,
        "modal_style": raw_res.get("modal_style", context_info["recommended_styles"]["modal_part"]),
        "breathy_text_clean": breathy_clean,
        "breathy_text_with_tag": breathy_tagged,
        "breathy_style": breathy_style,
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
    workers: int = 4,
) -> List[Dict[str, Any]]:
    """Build or load atomic sharded script pool across all 17 ATR sub-categories.

    Args:
        config_path: Path to ishi_contexts.yaml
        cache_dir: Directory where script_{id}.json files are stored
        scenarios_per_context: Number of scenarios per sub-category (default 3 -> ~51 scripts total)
        endpoint: Local LLM endpoint URL
        api_key: Local LLM API key
        model: Local LLM model identifier
        workers: Concurrency level for parallel generation (default: 4)

    Returns:
        List of script dictionary items sorted deterministically.
    """
    contexts = load_contexts(config_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_expected = len(contexts) * scenarios_per_context
    logger.info(f"Targeting script pool: {len(contexts)} sub-categories x {scenarios_per_context} = {total_expected} scripts (workers={workers}).")

    cached_scripts: List[Dict[str, Any]] = []
    tasks_to_generate: List[Tuple[Dict[str, Any], int, Path, str]] = []

    for ctx in contexts:
        ctx_id = ctx["id"]
        for s_idx in range(scenarios_per_context):
            script_id = f"script_{ctx_id}_{s_idx+1:02d}"
            shard_file = cache_dir / f"{script_id}.json"

            if shard_file.exists():
                try:
                    with open(shard_file, "r", encoding="utf-8") as f:
                        cached_item = json.load(f)
                    cached_scripts.append(cached_item)
                    continue
                except Exception as e:
                    logger.warning(f"Corrupted cache file {shard_file.name}, regenerating: {e}")

            tasks_to_generate.append((ctx, s_idx, shard_file, script_id))

    if cached_scripts:
        logger.info(f"Loaded {len(cached_scripts)} existing scripts from cache.")

    generated_scripts: List[Dict[str, Any]] = []

    if tasks_to_generate:
        logger.info(f"Generating {len(tasks_to_generate)} remaining scripts with {workers} worker threads...")
        completed_count = len(cached_scripts)
        lock = threading.Lock()

        def _worker_fn(task_info: Tuple[Dict[str, Any], int, Path, str]) -> Dict[str, Any]:
            c_info, idx, shard_path, s_id = task_info
            item = generate_single_script(
                context_info=c_info,
                scenario_idx=idx,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
            )
            save_atomic_script_shard(item, shard_path)
            return item

        if workers <= 1:
            for task in tasks_to_generate:
                item = _worker_fn(task)
                generated_scripts.append(item)
                completed_count += 1
                logger.info(f"[{completed_count}/{total_expected}] Generated and cached: {item['script_id']}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {executor.submit(_worker_fn, t): t for t in tasks_to_generate}
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        item = future.result()
                        generated_scripts.append(item)
                        with lock:
                            completed_count += 1
                            logger.info(f"[{completed_count}/{total_expected}] Generated and cached: {item['script_id']}")
                    except Exception as e:
                        logger.error(f"Failed generating script {task[3]}: {e}")
                        raise

    all_scripts = cached_scripts + generated_scripts
    all_scripts.sort(key=lambda x: x.get("script_id", ""))

    logger.info(f"Script pool ready: {len(all_scripts)} scripts available in {cache_dir}.")
    return all_scripts


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
        help="Number of scenarios to generate per sub-category (18 sub-categories * count = total).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for script generation (default: 4).",
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
        workers=args.workers,
    )
    print("\n" + "=" * 60)
    print(f"SCRIPT GENERATION COMPLETE!")
    print(f"- Total Scripts:   {len(pool)}")
    print(f"- Cache Directory: {args.cache_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
