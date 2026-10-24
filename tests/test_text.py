"""The add-on's BPE against Hugging Face's `tokenizers`, and the text cleanup.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The parity tests download the two pinned tokenizer.json files (a few MB) and
need the `tokenizers` package (the dev extra).
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chatterbox_addon import __main__ as addon, download, dsp, text  # noqa: E402

LINES = [
    ('en', "I'm terrified. I keep having nightmares about being chased by giant robotic claws of death."),
    ('pt', 'Estou apavorado! Tenho pesadelos que estou sendo perseguido por umas garras robóticas.'),
    ('fr', 'Nous devons partir avant le lever du soleil, sinon ils trouveront le vaisseau.'),
    ('de', 'Wir müssen vor Sonnenaufgang aufbrechen, sonst finden sie das Schiff. Größe!'),
    ('pl', 'Musimy wyjść przed świtem, inaczej znajdą statek. Żółć, gęś, źdźbło, Łódź.'),
    ('ru', 'Мы должны уйти до рассвета, иначе они найдут корабль. Ёжик в тумане.'),
    ('tr', 'Güneş doğmadan ayrılmalıyız, yoksa gemiyi bulacaklar. İstanbul.'),
    ('el', 'Πρέπει να φύγουμε πριν την ανατολή, αλλιώς θα βρουν το πλοίο.'),
    ('ar', 'يجب أن نغادر قبل شروق الشمس، وإلا سيجدون السفينة.'),
    ('hi', 'हमें सूर्योदय से पहले निकलना होगा, वरना वे जहाज़ ढूंढ लेंगे।'),
    ('ko', '해가 뜨기 전에 떠나야 해요, 그렇지 않으면 그들이 배를 찾을 거예요.'),
    ('sv', 'Vi måste åka före soluppgången, annars hittar de skeppet.'),
    ('en', "It's 3:45 p.m. — I'll be there in 2½ minutes… “quoted”; ‘single’ ² ¼ €5"),
    ('en', 'Oh, that is hilarious! [chuckle] Um anyway, [clear throat] how are you? 🙂 漢字'),
    ('en', "  multiple   spaces\tand\nnewlines  x'll y've 'd 'S 'T"),
]


def _tokenizer_file(key: str) -> Path:
    """The pinned tokenizer.json, downloaded (a few MB) if it is not there."""
    pin = next(p for p in addon.pins() if p['key'] == key and p['path'] == 'tokenizer.json')
    download.ensure([pin], lambda *_: None, lambda: False)
    return download.local_path(pin)


@pytest.fixture(scope='module')
def hf():
    return pytest.importorskip('tokenizers')


def test_multilingual_matches_tokenizers(hf):
    path = _tokenizer_file('multilingual')
    ours, theirs = text.MultilingualTokenizer(path), hf.Tokenizer.from_file(str(path))
    for language, line in LINES:
        prepared = text.prepare_multilingual(line, language)
        assert ours.encode_prepared(prepared) == theirs.encode(prepared, add_special_tokens=False).ids, line


def test_turbo_matches_tokenizers(hf):
    path = _tokenizer_file('turbo')
    ours, theirs = text.TurboTokenizer(path), hf.Tokenizer.from_file(str(path))
    for _, line in LINES:
        for variant in (line, text.punc_norm(line, multilingual=False)):
            assert ours.encode_raw(variant) == theirs.encode(variant, add_special_tokens=False).ids, variant


def test_decomposed_letters_fit_the_exports_vocabulary():
    path = _tokenizer_file('multilingual')
    tokenizer = text.MultilingualTokenizer(path)
    for language, line in LINES[:12]:
        ids = tokenizer.encode(line, language)
        assert text.MTL_UNK not in ids, (language, line)
        assert ids[0] == text.MTL_START_TEXT and ids[-1] == text.MTL_STOP_TEXT


def test_punc_norm_follows_upstream():
    assert text.punc_norm('hello world', multilingual=True) == 'Hello world.'
    assert text.punc_norm('wait... what: no; yes', multilingual=True) == 'Wait,  what, no,  yes.'
    assert text.punc_norm('wait... what', multilingual=False) == 'Wait... what.'
    assert text.punc_norm('他说。', multilingual=True) == '他说。'
    assert text.punc_norm('“Fine” — he said', multilingual=False) == '"Fine" - he said.'


def test_prepare_multilingual_lowercases_and_decomposes():
    prepared = text.prepare_multilingual('Żółć dla Łodzi', 'pl')
    assert prepared.startswith('[pl]') and '[SPACE]' in prepared and prepared == unicodedata.normalize('NFKD', prepared)
    assert 'ż' not in prepared and 'ż' in prepared
    korean = text.prepare_multilingual('한국', 'ko')
    assert all(not ('가' <= c <= '힯') for c in korean)


def test_loudness_and_reference_shaping():
    rate = 24000
    tone = (0.1 * np.sin(np.arange(rate * 3) * 2 * np.pi * 440 / rate)).astype(np.float32)
    assert -24 < dsp.loudness(tone, rate) < -23
    assert abs(dsp.loudness(dsp.normalize_loudness(tone, rate), rate) - dsp.TARGET_LUFS) < 0.01
    looped = dsp.loop(tone, rate)
    assert len(looped) >= dsp.MIN_SECONDS * rate
    long = np.tile(tone, 5)
    assert len(dsp.cut(long, rate)) <= dsp.MAX_SECONDS * rate
    assert dsp.loudness(np.zeros(rate, dtype=np.float32), rate) == float('-inf')
