#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script generator for Japanese breathy voice synthesis.

Invokes gemini-3.8-flash-high at http://127.0.0.1:8045 to generate structured
conversational turns with segmented modal and breathy parts.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("script_generator")

DEFAULT_TEXT_ENDPOINT = "http://127.0.0.1:8045"
DEFAULT_TEXT_KEY = "ENV_LOCAL_LLM_KEY_PLACEHOLDER"
DEFAULT_TEXT_MODEL = "gemini-3.8-flash-high"

VOICE_ROTATION = ["Kore", "Aoede", "Puck", "Fenrir"]


def load_contexts(config_path: Path) -> List[Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("contexts", [])


def clean_transcript_for_tifa(text: str) -> str:
    """Remove inline angle tags like <breath>, <sigh> for phonetic alignment."""
    cleaned = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple whitespaces or punctuation glitches
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def call_llm_json(prompt: str, endpoint: str = DEFAULT_TEXT_ENDPOINT, api_key: str = DEFAULT_TEXT_KEY, model: str = DEFAULT_TEXT_MODEL) -> Dict[str, Any]:
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
                    "You are an expert Japanese computational phonetician and dialogue director. "
                    "Your task is to generate natural conversational Japanese utterances for speech synthesis evaluation. "
                    "Always output strict valid JSON only, without any markdown formatting, backticks, or comments."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=45) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        content = res["choices"][0]["message"]["content"].strip()

    # Extract JSON block if wrapped in markdown
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def generate_single_script(
    context_info: Dict[str, Any],
    scenario_idx: int,
    voice: str,
    sample_id: str,
    endpoint: str = DEFAULT_TEXT_ENDPOINT,
    api_key: str = DEFAULT_TEXT_KEY,
    model: str = DEFAULT_TEXT_MODEL,
) -> Dict[str, Any]:
    scenario = context_info["seed_scenarios"][scenario_idx % len(context_info["seed_scenarios"])]
    
    prompt = f"""
日本語の音声対話合成評価用の台詞を1件作成してください。

【対象の副言語的気声（Breathy Voice）コンテキスト】
- カテゴリID: {context_info['id']}
- 名称: {context_info['name']}
- 背景説明: {context_info['description']}
- 状況ヒント: {scenario['prompt_hint']}
- 推奨タグ: {context_info.get('tag_recommendation', '<breath>')}

【要件】
1. 話者Aの短い導入発話（Context）と、話者Bの応答発話を作成してください。
2. 音声合成対象は【話者Bの発話のみ】です。
3. 発話Bの長さは自然な会話として3秒〜7秒程度（平仮名換算で20字〜45字程度）にしてください（短すぎないこと）。
4. 発話Bは必ず以下の2つのパートに論理的に分割してください：
   - modal_text: 導入部の通常の常態音（Modal Voice）で話す部分（例: 相槌、前置きなど、全体の30%〜50%）
   - breathy_text: 感情や意図が強く込められ、気声（Breathy Voice）で話す核心部分（全体の50%〜70%）
     ※重要：気声（Breathy Voice）は声帯の振動（音程・ピッチ）を維持したまま息が漏れる発声です。ささやき声（Whisper）ではありません。styleにwhisperやwhisper-likeという単語は絶対に含めないでください。
5. breathy_textの先頭には、息の漏れを表すタグ {context_info.get('tag_recommendation', '<breath>')} を挿入してください。
6. 出力は以下のJSONフォーマットのみを返してください：

{{
  "context_lead_in": "話者Aの台詞",
  "modal_text": "話者Bの前半の台詞（通常の平仮名/漢字、タグなし）",
  "modal_style": "{context_info['recommended_styles']['modal_part']}",
  "breathy_text_clean": "話者Bの後半の台詞（タグなしの純粋なテキスト）",
  "breathy_text_with_tag": "{context_info.get('tag_recommendation', '<breath>')} 話者Bの後半の台詞",
  "breathy_style": "{context_info['recommended_styles']['breathy_part']}"
}}
"""
    logger.info(f"Generating script for {sample_id} ({context_info['id']} #{scenario_idx})...")
    raw_res = call_llm_json(prompt, endpoint=endpoint, api_key=api_key, model=model)

    modal_text = raw_res["modal_text"].strip()
    breathy_clean = raw_res["breathy_text_clean"].strip()
    breathy_tagged = raw_res.get("breathy_text_with_tag", f"{context_info.get('tag_recommendation', '<breath>')} {breathy_clean}").strip()
    clean_full = clean_transcript_for_tifa(f"{modal_text} {breathy_clean}")

    parts = [
        {
            "text": modal_text,
            "speech_metadata": {
                "speaker": voice,
                "style": raw_res.get("modal_style", context_info["recommended_styles"]["modal_part"]),
            },
        },
        {
            "text": breathy_tagged,
            "speech_metadata": {
                "speaker": voice,
                "style": raw_res.get("breathy_style", context_info["recommended_styles"]["breathy_part"]),
            },
        },
    ]

    return {
        "sample_id": sample_id,
        "context_id": context_info["id"],
        "context_name": context_info["name"],
        "voice": voice,
        "context_lead_in": raw_res.get("context_lead_in", ""),
        "modal_text": modal_text,
        "breathy_text_clean": breathy_clean,
        "clean_full_text": clean_full,
        "tts_parts": parts,
    }


def generate_smoke_test_scripts(
    config_path: Path,
    output_json: Path,
    endpoint: str = DEFAULT_TEXT_ENDPOINT,
    api_key: str = DEFAULT_TEXT_KEY,
    model: str = DEFAULT_TEXT_MODEL,
) -> List[Dict[str, Any]]:
    contexts = load_contexts(config_path)
    scripts: List[Dict[str, Any]] = []

    count = 0
    for ctx in contexts:
        for idx in range(2):  # 2 samples per context -> 10 samples total
            voice = VOICE_ROTATION[count % len(VOICE_ROTATION)]
            sample_id = f"smoke_{count+1:02d}_{ctx['id']}_{voice.lower()}"
            item = generate_single_script(
                context_info=ctx,
                scenario_idx=idx,
                voice=voice,
                sample_id=sample_id,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
            )
            scripts.append(item)
            count += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scripts, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully generated {len(scripts)} scripts saved to {output_json}")
    return scripts


if __name__ == "__main__":
    cfg_file = Path(__file__).parent / "configs" / "ishi_contexts.yaml"
    out_file = Path("exp_results/tts_smoke_test/smoke_scripts.json")
    res = generate_smoke_test_scripts(cfg_file, out_file)
    print(f"Generated {len(res)} scripts.")
