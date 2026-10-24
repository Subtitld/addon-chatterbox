"""The voices the add-on offers, and the languages each model speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Chatterbox has no voices of its own: every line clones a reference clip.

- `chatterbox-clone` clones the speaker from the video: the host hands over a
  clip of their speech (or Configure names one).
- The presets clone a clip that ships with the add-on: audiobook readers from
  LibriVox, whose recordings are in the public domain, taken from LibriTTS-R
  (CC-BY-4.0) for English and from the LibriVox originals for the rest (the
  same clips as Subtitld's Sopro add-on). presets/README.md credits each one.
  A preset speaks every language; its own language only sets its accent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CLONE = 'chatterbox-clone'

# Multilingual's 23 languages, less Hebrew (upstream adds vowel points with a
# separate model first), Japanese (it turns kanji into kana with pykakasi,
# which is GPL) and Chinese (garbled in our test of the export).
LANGUAGE_NAMES = {
    'ar': 'Arabic', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English',
    'es': 'Spanish', 'fi': 'Finnish', 'fr': 'French', 'hi': 'Hindi', 'it': 'Italian',
    'ko': 'Korean', 'ms': 'Malay', 'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish',
    'pt': 'Portuguese', 'ru': 'Russian', 'sv': 'Swedish', 'sw': 'Swahili', 'tr': 'Turkish',
}
LANGUAGES = tuple(LANGUAGE_NAMES)
TURBO_LANGUAGES = ('en',)
# Tags Subtitld or a user may send for a language under another code.
ALIASES = {'nb': 'no', 'nn': 'no', 'zsm': 'ms', 'swh': 'sw'}

PRESETS_DIR = Path(__file__).resolve().parent / 'presets'


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    clip: str = ''   # file name in presets/, empty for the clone


@lru_cache(maxsize=1)
def presets() -> tuple[Voice, ...]:
    index = json.loads((PRESETS_DIR / 'presets.json').read_text(encoding='utf-8'))
    return tuple(Voice(p['id'], p['display_name'], p['clip']) for p in index['presets'])


def all_voices() -> tuple[Voice, ...]:
    return (Voice(CLONE, "Speaker's own voice (cloned from the video)"), *presets())


def by_id(voice_id: str) -> Voice | None:
    return next((v for v in all_voices() if v.id == voice_id), None)


def clip_path(voice: Voice) -> Path:
    return PRESETS_DIR / voice.clip


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in all_voices():
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name, 'sample_rate': 24000}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out


def primary_language(tag) -> str:
    """`pt-BR` -> `pt`, `nb` -> `no`; '' when there is none."""
    primary = str(tag or '').strip().lower().replace('_', '-').split('-', 1)[0]
    return ALIASES.get(primary, primary)
