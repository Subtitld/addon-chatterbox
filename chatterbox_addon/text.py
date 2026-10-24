"""Text front end: Chatterbox's punctuation cleanup and its two tokenizers.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Upstream (resemble-ai/chatterbox, MIT) tokenizes with Hugging Face's
`tokenizers` library. The add-on reads the same tokenizer.json files with the
small BPE below instead, so no native library (and no licence notice for its
statically linked crates) ships for something this simple.
tests/test_text.py checks both against `tokenizers` on many lines.

- Multilingual: a character-level BPE (265 merges). Upstream lowercases the
  line and decomposes it (NFKD) first, puts the language tag in front and
  turns spaces into a [SPACE] token. The decomposition matters: the export's
  vocabulary has no "ś" or "й", only "s", "и" and the combining accents, so
  a line that skipped it would lose those letters.
- Turbo: GPT-2's byte-level BPE, plus the paralinguistic tags ([laugh],
  [sigh], ...) as whole tokens.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

# The Multilingual export's text embedding has 2352 rows (the first
# checkpoint's vocabulary); its tokenizer.json lists a later, larger one.
MTL_TEXT_VOCAB = 2352
MTL_UNK = 1
MTL_START_TEXT = 255
MTL_STOP_TEXT = 0
MTL_SPACE = '[SPACE]'


def punc_norm(text: str, multilingual: bool) -> str:
    """Upstream's punc_norm (mtl_tts.py and tts_turbo.py differ slightly)."""
    text = ' '.join(text.split())
    if not text:
        return text
    if text[0].islower():
        text = text[0].upper() + text[1:]
    replace = [('…', ', '), (':', ','), ('—', '-'), ('–', '-'), (' ,', ','),
               ('“', '"'), ('”', '"'), ('‘', "'"), ('’', "'")]
    if multilingual:
        replace = [('...', ', '), ('…', ', '), (':', ','), (' - ', ', '), (';', ', ')] + replace[2:]
    for old, new in replace:
        text = text.replace(old, new)
    text = text.rstrip(' ')
    enders = ('.', '!', '?', '-', ',') + (('、', '，', '。', '？', '！') if multilingual else ())
    if not text.endswith(enders):
        text += '.'
    return text


def korean_normalize(text: str) -> str:
    """Hangul syllables to conjoining jamo, as upstream does for Korean."""
    out = []
    for char in text:
        if '가' <= char <= '힯':
            base = ord(char) - 0xAC00
            out.append(chr(0x1100 + base // 588) + chr(0x1161 + (base % 588) // 28)
                       + (chr(0x11A7 + base % 28) if base % 28 else ''))
        else:
            out.append(char)
    return ''.join(out).strip()


def prepare_multilingual(text: str, language: str) -> str:
    """What upstream's MTLTokenizer.encode does before the BPE itself."""
    text = unicodedata.normalize('NFKD', punc_norm(text, multilingual=True).lower())
    if language == 'ko':
        text = korean_normalize(text)
    return f'[{language}]{text}'.replace(' ', MTL_SPACE)


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------
class _BPE:
    def __init__(self, path: Path):
        spec = json.loads(Path(path).read_text(encoding='utf-8'))
        model = spec['model']
        if model.get('type') != 'BPE':
            raise ValueError(f'{path}: not a BPE tokenizer')
        self.vocab: dict[str, int] = model['vocab']
        merges = [tuple(m.split(' ', 1)) if isinstance(m, str) else tuple(m) for m in model['merges']]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.unk = self.vocab.get(model.get('unk_token') or '')
        self.added = {t['content']: t['id'] for t in spec.get('added_tokens', [])}
        # Longest first, so "[clear throat]" wins over a shorter prefix.
        names = sorted(self.added, key=len, reverse=True)
        self._split = re.compile('(' + '|'.join(re.escape(n) for n in names) + ')') if names else None
        self._cache: dict[str, list[str]] = {}

    def _merge(self, word: str) -> list[str]:
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            best = min(range(len(parts) - 1),
                       key=lambda i: self.ranks.get((parts[i], parts[i + 1]), sys.maxsize))
            if (parts[best], parts[best + 1]) not in self.ranks:
                break
            pair = (parts[best], parts[best + 1])
            merged, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        if len(self._cache) < 50000:
            self._cache[word] = parts
        return parts

    def _pieces(self, text: str):
        """(is_added_token, text) runs, the added tokens matched first."""
        if self._split is None:
            yield False, text
            return
        for i, piece in enumerate(self._split.split(text)):
            if piece:
                yield i % 2 == 1, piece

    def _word_ids(self, word: str) -> list[int]:
        out = []
        for token in self._merge(word):
            token_id = self.vocab.get(token)
            if token_id is None:
                if self.unk is None:
                    raise ValueError(f'no token for {token!r}')
                token_id = self.unk
            out.append(token_id)
        return out


class MultilingualTokenizer(_BPE):
    """tokenizer.json of onnx-community/chatterbox-multilingual-ONNX."""

    _words = re.compile(r'\w+|[^\w\s]+')   # the Whitespace pre-tokenizer

    def encode_prepared(self, text: str) -> list[int]:
        """Ids of a line prepare_multilingual() returned, without specials."""
        ids: list[int] = []
        for added, piece in self._pieces(text):
            if added:
                ids.append(self.added[piece])
                continue
            for word in self._words.findall(piece):
                ids.extend(self._word_ids(word))
        return ids

    def encode(self, text: str, language: str) -> list[int]:
        """[start of text] + the line + [end of text], ready for the model.

        Ids past the export's vocabulary (letters the later tokenizer added)
        become [UNK]; prepare_multilingual() decomposes most of them away.
        """
        ids = self.encode_prepared(prepare_multilingual(text, language))
        ids = [i if i < MTL_TEXT_VOCAB else MTL_UNK for i in ids]
        return [MTL_START_TEXT, *ids, MTL_STOP_TEXT]


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """GPT-2's printable stand-in for each byte."""
    keep = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars)))


@lru_cache(maxsize=1)
def _gpt2_pattern() -> re.Pattern:
    r"""GPT-2's pre-tokenizer regex. Python's re has no \p{L} or \p{N}, so the
    two classes are built from the Unicode database (once, about 0.3 s)."""
    letters, numbers = [], []
    for cp in range(sys.maxunicode + 1):
        category = unicodedata.category(chr(cp))
        if category[0] == 'L':
            letters.append(cp)
        elif category[0] == 'N':
            numbers.append(cp)

    def char_class(points: list[int]) -> str:
        out, start, prev = [], points[0], points[0]
        for cp in points[1:] + [-1]:
            if cp == prev + 1:
                prev = cp
                continue
            out.append(re.escape(chr(start)) if start == prev else f'{re.escape(chr(start))}-{re.escape(chr(prev))}')
            start = prev = cp
        return ''.join(out)

    L, N = char_class(letters), char_class(numbers)
    return re.compile(rf"""'s|'t|'re|'ve|'m|'ll|'d| ?[{L}]+| ?[{N}]+| ?[^\s{L}{N}]+|\s+(?!\S)|\s+""")


class TurboTokenizer(_BPE):
    """tokenizer.json of ResembleAI/chatterbox-turbo-ONNX (GPT-2's BPE)."""

    END_OF_TEXT = '<|endoftext|>'

    def encode_raw(self, text: str) -> list[int]:
        pattern, to_unicode = _gpt2_pattern(), _byte_to_unicode()
        ids: list[int] = []
        for added, piece in self._pieces(text):
            if added:
                ids.append(self.added[piece])
                continue
            for word in pattern.findall(piece):
                ids.extend(self._word_ids(''.join(to_unicode[b] for b in word.encode('utf-8'))))
        return ids

    def encode(self, text: str) -> list[int]:
        """The line + two end-of-text tokens, as the export's template has it:
        the embedding graph reads the last one as the start of speech."""
        end = self.added[self.END_OF_TEXT]
        return [*self.encode_raw(punc_norm(text, multilingual=False)), end, end]
