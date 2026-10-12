import re

def sanitize_transcript_for_tifa(text: str) -> str:
    """Sanitize Japanese transcript text specifically for TIFA G2P alignment.
    
    Removes bracket punctuation that fugashi/TIFA fails to strip, expands
    English abbreviations, and normalizes colloquial small-vowel contractions.
    """
    # 1. Remove Japanese brackets and quotes
    text = re.sub(r"[「」『』【】（）()\"'“”]", " ", text)
    # 2. Expand common Latin abbreviations
    text = re.sub(r"\bSNS\b", "エスエヌエス", text, flags=re.IGNORECASE)
    # 3. Normalize colloquial small-vowel contractions
    text = text.replace("ぱぁ", "ぱあ")
    text = text.replace("だぁ", "だあ")
    text = text.replace("カァ", "カー")
    text = text.replace("ワァ", "ワー")
    text = text.replace("パァ", "パー")
    # 4. Normalize any katakana + small vowel + chouonpu
    text = re.sub(r"([カサタナハマヤラワガザダバパ])ァー", r"\1ー", text)
    text = re.sub(r"([キシチニヒミリギジヂビピ])ィー", r"\1ー", text)
    text = re.sub(r"([クスツヌフムユルグズヅブプ])ゥー", r"\1ー", text)
    text = re.sub(r"([ケセテネヘメレゲゼデベペ])ェー", r"\1ー", text)
    text = re.sub(r"([コソトノホモヨロゴゾドボポ])ォー", r"\1ー", text)
    # 5. Clean whitespace
    return re.sub(r"\s+", " ", text).strip()
