"""manifest.json, pins.json, the presets and the code agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chatterbox_addon import ADDON_ID, __version__, audio, voices  # noqa: E402
from chatterbox_addon import __main__ as addon  # noqa: E402
from chatterbox_addon.engine import FILES, graph_files  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
PINS = json.loads((REPO_ROOT / 'chatterbox_addon' / 'pins.json').read_text(encoding='utf-8'))
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True
    assert MANIFEST['compute'] == 'cpu'
    fields = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert set(fields) == {'model', 'threads', 'expressiveness', 'voice_ref_audio'}
    assert [o['value'] for o in fields['model']['options']] == [m['id'] for m in MANIFEST['models']] == list(FILES)


def test_models_match_the_pins():
    for model in MANIFEST['models']:
        pinned = addon.pins(model['id'])
        assert len(pinned) == len(FILES[model['id']])
        assert model['size_mb'] == round(sum(p['size'] for p in pinned) / 1e6)
        assert model['revision'] == PINS[model['id']]['revision'] and model['repo'] == PINS[model['id']]['repo']
    assert MANIFEST['disk_required_mb'] >= sum(m['size_mb'] for m in MANIFEST['models'])


def test_every_pin_is_a_checked_file():
    for pin in addon.pins():
        assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
        assert re.fullmatch(r'[0-9a-f]{40}', pin['revision']), pin
    # Each graph's weights come with it.
    paths = {(p['key'], p['path']) for p in addon.pins()}
    for key, path in paths:
        if path.endswith('.onnx'):
            assert (key, path + '_data') in paths


def test_graph_files_name_each_role(tmp_path):
    for model in FILES:
        roles = graph_files(model, lambda key, path: tmp_path / key / path)
        expected = {'tokenizer', 'speech_encoder', 'embed_tokens', 'language_model', 'conditional_decoder'}
        assert set(roles) == expected
    assert addon.model_files('multilingual')['conditional_decoder'] == addon.model_files('turbo')['conditional_decoder']


def test_every_preset_is_a_clip_of_one_voice():
    presets = voices.presets()
    assert len(presets) == 8 and len({p.id for p in presets}) == 8
    assert all(p.id.startswith('chatterbox-') for p in presets)
    index = json.loads((voices.PRESETS_DIR / 'presets.json').read_text(encoding='utf-8'))['presets']
    for entry, voice in zip(index, presets):
        samples, rate = audio.read_wav(voices.clip_path(voice))
        assert rate == 24000 and 9 < len(samples) / rate < 16, voice.id
        assert entry['language'] in voices.LANGUAGES
        assert entry['source'] in ('LibriTTS-R', 'LibriVox via Multilingual LibriSpeech')
    credits = (voices.PRESETS_DIR / 'README.md').read_text(encoding='utf-8')
    assert all(f'`{v.clip}`' in credits for v in presets)
    assert (voices.PRESETS_DIR / 'CC-BY-4.0.txt').is_file()


@pytest.mark.parametrize('tag, expected', [('pt-BR', 'pt'), ('nb', 'no'), ('EN_us', 'en'), ('', '')])
def test_language_tags(tag, expected):
    assert voices.primary_language(tag) == expected
