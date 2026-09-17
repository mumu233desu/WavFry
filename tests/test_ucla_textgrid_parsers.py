# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from scripts.convert_textgrid_to_breathy_cache import (
    parse_ucla_mazatec,
    parse_ucla_gujarati,
    parse_ucla_zapotec,
    parse_ucla_hmong,
    parse_ucla_miao,
    DATASET_PARSERS,
)

def test_dataset_parsers_registry():
    expected = {'ucla_mazatec', 'ucla_gujarati', 'ucla_zapotec', 'ucla_hmong', 'ucla_miao'}
    assert set(DATASET_PARSERS.keys()) == expected

def test_parse_ucla_mazatec():
    tiers = {
        'phonation': [(0.1, 0.5, 'breathy'), (0.6, 1.0, 'modal'), (1.1, 1.5, 'creaky')],
        'vowel': [(0.1, 0.5, 'a'), (0.6, 1.0, 'o'), (1.1, 1.5, 'u')],
    }
    ann = parse_ucla_mazatec(tiers, 'hwa2_F1', Path('hwa2_F1.wav'), Path('hwa2_F1.TextGrid'))
    assert ann['0.1000|0.5000|a'] == 'breathy'
    assert ann['0.6000|1.0000|o'] == 'non_breathy'
    assert ann['1.1000|1.5000|u'] == 'creaky'

def test_parse_ucla_gujarati():
    tiers = {
        'vowels': [(0.1, 0.4, 'a='), (0.5, 0.8, 'aH'), (0.9, 1.2, 'a')],
    }
    ann = parse_ucla_gujarati(tiers, 'sample_Audio', Path('sample_Audio.wav'), Path('sample_Audio.TextGrid'))
    assert ann['0.1000|0.4000|a'] == 'breathy'
    assert ann['0.5000|0.8000|aH'] == 'uncertain'
    assert ann['0.9000|1.2000|a'] == 'non_breathy'

def test_parse_ucla_zapotec():
    tiers = {
        'vowels': [(0.2, 0.6, 'aB14'), (0.7, 1.1, 'aC31'), (1.2, 1.5, 'aM1')],
    }
    ann = parse_ucla_zapotec(tiers, 'speaker1M_aB14', Path('speaker1M_aB14.wav'), Path('speaker1M_aB14.TextGrid'))
    assert ann['0.2000|0.6000|a'] == 'breathy'
    assert ann['0.7000|1.1000|a'] == 'creaky'
    assert ann['1.2000|1.5000|a'] == 'non_breathy'

    # Fallback to filename when TextGrid interval is backtick
    tiers_irregular = {'vowels': [(0.1, 0.5, "`")]}
    ann_irreg = parse_ucla_zapotec(tiers_irregular, 'speaker2M_aB14', Path('speaker2M_aB14.wav'), Path('speaker2M_aB14.TextGrid'))
    assert ann_irreg['0.1000|0.5000|V'] == 'breathy'

def test_parse_ucla_hmong():
    tiers = {'vowel': [(0.1, 0.5, 'a')]}
    # -g breathy
    ann_g = parse_ucla_hmong(tiers, '1-caag_1_-g_Audio', Path('1-caag_1_-g_Audio.wav'), Path('1-caag_1_-g_Audio.TextGrid'))
    assert ann_g['0.1000|0.5000|a'] == 'breathy'

    # -m creaky
    ann_m = parse_ucla_hmong(tiers, '1-caam-g_Audio', Path('1-caam-g_Audio.wav'), Path('1-caam-g_Audio.TextGrid'))
    assert ann_m['0.1000|0.5000|a'] == 'creaky'

    # modal
    ann_modal = parse_ucla_hmong(tiers, '1-caa-w_Audio', Path('1-caa-w_Audio.wav'), Path('1-caa-w_Audio.TextGrid'))
    assert ann_modal['0.1000|0.5000|a'] == 'non_breathy'

def test_parse_ucla_miao():
    tiers = {
        'seg': [
            (0.0, 0.1, 'p'),
            (0.1, 0.5, 'ta33_01'),
            (0.5, 0.6, 't'),
            (0.6, 1.0, 'pu11_01'),
            (1.0, 1.4, 'a_44'),
            (1.4, 1.8, 'se55_01'),
        ]
    }
    ann = parse_ucla_miao(tiers, 'sample_01', Path('sample_01.wav'), Path('sample_01.TextGrid'))
    # Consonant p and t must be skipped
    assert not any('p' == k.split('|')[2] for k in ann)
    assert not any('t' == k.split('|')[2] for k in ann)
    assert ann['0.1000|0.5000|ta'] == 'breathy'     # tone 33
    assert ann['0.6000|1.0000|pu'] == 'creaky'      # tone 11
    assert ann['1.0000|1.4000|a'] == 'non_breathy'   # tone 44
    assert ann['1.4000|1.8000|se'] == 'uncertain'   # tone 55 (tense)
