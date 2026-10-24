"""Chatterbox on ONNX Runtime: clone a reference voice, speak a line.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The graphs
----------
Both exports (MIT) split the model into four graphs; the loop between them is
upstream's model-card example, ported to the add-on:

- speech_encoder: the 24 kHz reference → the speaker conditioning for the
  language model, the clip's speech tokens, a speaker vector and its mel
  features for the decoder.
- embed_tokens: token ids → embeddings.
- language_model: a Llama (Multilingual, 30 layers) or GPT-2 (Turbo, 24
  layers) with a KV cache, from embeddings to the next speech token's logits.
  The 4-bit builds: the 8-bit ones run about seven times slower on a CPU.
- conditional_decoder: speech tokens + the speaker → the waveform.

Both models use Turbo's decoder, the flow decoder distilled to two steps
(upstream's s3gen_meanflow). The Multilingual export ships the ten-step one,
which made the decoder take three quarters of the time; Turbo's decodes the
same speech tokens about four times faster and, in our test, sounded as close
to the speaker or closer.

Decoding
--------
Upstream samples (temperature 0.8) and, for Multilingual, uses
classifier-free guidance, with an attention-based monitor that stops a line
that runs on. The monitor needs attention maps the export does not give, and
without it sampling ran on in a third of our test lines, so the add-on
decodes greedily, with upstream's repetition penalty of 1.2. A line can still
get stuck repeating one token: ten in a row ends the attempt, and the line is
tried again with guidance (Multilingual) or sampling, before the last attempt
is kept as it is.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from chatterbox_addon import dsp, onnxdata
from chatterbox_addon.text import MultilingualTokenizer, TurboTokenizer

log = logging.getLogger('chatterbox_addon.engine')

SAMPLE_RATE = 24000
TOKENS_PER_SECOND = 25
START_SPEECH = 6561
STOP_SPEECH = 6562
SILENCE = 4299
REPETITION_PENALTY = 1.2
STUCK_RUN = 10            # the same token this many times in a row: stuck
MAX_TOKENS = 1000         # upstream's cap, 40 seconds

MODELS = ('multilingual', 'turbo')

# The files each model reads, by (repository key in pins.json, path).
FILES = {
    'multilingual': [
        ('multilingual', 'tokenizer.json'),
        ('multilingual', 'onnx/speech_encoder.onnx'), ('multilingual', 'onnx/speech_encoder.onnx_data'),
        ('multilingual', 'onnx/embed_tokens.onnx'), ('multilingual', 'onnx/embed_tokens.onnx_data'),
        ('multilingual', 'onnx/language_model_q4.onnx'), ('multilingual', 'onnx/language_model_q4.onnx_data'),
        ('turbo', 'onnx/conditional_decoder_q4.onnx'), ('turbo', 'onnx/conditional_decoder_q4.onnx_data'),
    ],
    'turbo': [
        ('turbo', 'tokenizer.json'),
        ('turbo', 'onnx/speech_encoder_q4.onnx'), ('turbo', 'onnx/speech_encoder_q4.onnx_data'),
        ('turbo', 'onnx/embed_tokens_q4.onnx'), ('turbo', 'onnx/embed_tokens_q4.onnx_data'),
        ('turbo', 'onnx/language_model_q4.onnx'), ('turbo', 'onnx/language_model_q4.onnx_data'),
        ('turbo', 'onnx/conditional_decoder_q4.onnx'), ('turbo', 'onnx/conditional_decoder_q4.onnx_data'),
    ],
}


class Cancelled(Exception):
    """The host cancelled the line."""


@dataclass
class Reference:
    conditioning: np.ndarray      # (1, n, 1024) for the language model
    prompt_tokens: np.ndarray     # (1, m) int64, the clip's speech tokens
    speaker: np.ndarray           # (1, 192)
    features: np.ndarray          # (1, frames, 80)


@dataclass
class Attempt:
    guidance: float
    sample: bool


def default_threads() -> int:
    """Half the logical processors, at most 8: past that it got no faster."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def _penalize(logits: np.ndarray, tokens: list[int]) -> np.ndarray:
    """Upstream's repetition penalty, on every token generated so far."""
    seen = np.unique(np.asarray(tokens))
    scores = logits[seen]
    logits[seen] = np.where(scores < 0, scores * REPETITION_PENALTY, scores / REPETITION_PENALTY)
    return logits


def _sample(logits: np.ndarray, rng: np.random.Generator, temperature: float = 0.8,
            top_k: int = 1000, top_p: float = 0.95, min_p: float = 0.0) -> int:
    scores = logits / max(temperature, 1e-3)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    if min_p > 0:
        probs[probs < min_p * probs.max()] = 0
    order = np.argsort(-probs)
    if top_k and top_k < len(order):
        probs[order[top_k:]] = 0
        order = order[:top_k]
    if top_p < 1:
        cumulative = np.cumsum(probs[order])
        probs[order[(cumulative - probs[order]) > top_p]] = 0
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


class _Chatterbox:
    """What both models share: the reference, the loop and the decoder."""

    name = ''
    attempts: tuple[Attempt, ...] = ()
    sampling: dict = {}
    normalize_reference = False   # to -27 LUFS, as upstream's Turbo does

    def __init__(self, files: dict[str, Path], threads: int):
        self.threads = threads
        self.encoder = _session(files['speech_encoder'], threads)
        self.language_model = _session(files['language_model'], threads)
        self.decoder = _session(files['conditional_decoder'], threads)
        inputs = self.language_model.get_inputs()
        self._past_names = [i.name for i in inputs if i.name.startswith('past_key_values')]
        self._has_positions = any(i.name == 'position_ids' for i in inputs)

    # -- the reference -----------------------------------------------------
    def prepare_reference(self, samples: np.ndarray) -> Reference:
        """A 24 kHz mono clip of the voice → what every line in it needs."""
        clip = dsp.loop(dsp.cut(np.asarray(samples, dtype=np.float32), SAMPLE_RATE), SAMPLE_RATE)
        if self.normalize_reference:
            clip = dsp.normalize_loudness(clip, SAMPLE_RATE)
        conditioning, tokens, speaker, features = self.encoder.run(None, {'audio_values': clip[None]})
        return Reference(conditioning, tokens.astype(np.int64), speaker, features)

    # -- one line ----------------------------------------------------------
    def _prompt(self, text: str, language: str, reference: Reference, guidance: bool) -> np.ndarray:
        raise NotImplementedError

    def _embed_speech(self, token: int, step: int) -> np.ndarray:
        raise NotImplementedError

    def _generate(self, prompt: np.ndarray, attempt: Attempt, seed: int, limit: int,
                  progress: Callable[[float], None], cancelled: Callable[[], bool]) -> tuple[list[int], bool]:
        """Speech tokens, and whether the line ended on its own."""
        rng = np.random.default_rng(seed)
        batch = prompt.shape[0]
        embeds = prompt
        mask = np.ones((batch, prompt.shape[1]), dtype=np.int64)
        positions = np.arange(prompt.shape[1], dtype=np.int64)[None].repeat(batch, axis=0)
        past = {name: np.zeros((batch, 16, 0, 64), dtype=np.float32) for name in self._past_names}
        generated = [START_SPEECH]
        for step in range(limit):
            if cancelled():
                raise Cancelled()
            feed = {'inputs_embeds': embeds, 'attention_mask': mask, **past}
            if self._has_positions:
                feed['position_ids'] = positions
            logits, *present = self.language_model.run(None, feed)
            scores = logits[:, -1, :].astype(np.float64)
            if batch == 2:
                scores = scores[0] + attempt.guidance * (scores[0] - scores[1])
            else:
                scores = scores[0]
            scores = _penalize(scores, generated)
            token = _sample(scores, rng, **self.sampling) if attempt.sample else int(np.argmax(scores))
            if token == STOP_SPEECH:
                return generated[1:], True
            generated.append(token)
            if len(generated) > STUCK_RUN and len(set(generated[-STUCK_RUN:])) == 1:
                return generated[1:-STUCK_RUN + 2], False
            progress(min(1.0, step / max(1, limit * 0.4)))
            embeds = np.repeat(self._embed_speech(token, step + 1), batch, axis=0)
            mask = np.concatenate([mask, np.ones((batch, 1), dtype=np.int64)], axis=1)
            positions = positions[:, -1:] + 1
            past = dict(zip(self._past_names, present))
        return generated[1:], False

    def synthesize(self, text: str, reference: Reference, *, language: str = 'en', seed: int = 42,
                   progress: Callable[[float], None] = lambda f: None,
                   cancelled: Callable[[], bool] = lambda: False) -> np.ndarray:
        """24 kHz mono float32 samples of `text` in the reference's voice."""
        limit = min(MAX_TOKENS, 80 + 6 * len(text))
        tokens: list[int] = []
        for number, attempt in enumerate(self.attempts):
            prompt = self._prompt(text, language, reference, guidance=attempt.guidance > 0)

            tokens, finished = self._generate(prompt, attempt, seed + number, limit,
                                              lambda fraction: progress(0.9 * fraction), cancelled)
            if finished:
                break
            log.warning('%s: the line got stuck after %d tokens (attempt %d of %d)',
                        self.name, len(tokens), number + 1, len(self.attempts))
        tokens = [t for t in tokens if t < START_SPEECH]
        speech = np.concatenate([reference.prompt_tokens,
                                 np.asarray([tokens + [SILENCE] * 3], dtype=np.int64)], axis=1)
        if cancelled():
            raise Cancelled()
        # The decoder starts its flow from noise the graph draws itself
        # (RandomNormalLike, with no seed input), so the samples differ a
        # little from run to run; the speech tokens, and so the words, the
        # timing and the voice, do not.
        wav = self.decoder.run(None, {'speech_tokens': speech, 'speaker_embeddings': reference.speaker,
                                      'speaker_features': reference.features})[0][0]
        return np.asarray(wav, dtype=np.float32)


class Multilingual(_Chatterbox):
    """onnx-community/chatterbox-multilingual-ONNX, with Turbo's decoder."""

    name = 'multilingual'
    attempts = (Attempt(0.0, False), Attempt(0.5, False), Attempt(0.5, True))
    sampling = {'temperature': 0.8, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.05}

    def __init__(self, files: dict[str, Path], threads: int, exaggeration: float = 0.5):
        super().__init__(files, threads)
        self.tokenizer = MultilingualTokenizer(files['tokenizer'])
        tables = onnxdata.initializers(files['embed_tokens'])
        self.text_embedding = tables['text_emb.weight']
        self.text_positions = tables['text_pos_emb.weight']
        self.speech_embedding = tables['speech_emb.weight']
        self.speech_positions = tables['speech_pos_emb.weight']
        # emotion_adv_fc: a 1 → 1024 projection without a bias, the only
        # (1, 1024) initializer in the graph.
        self.emotion = next(v for v in tables.values() if v.shape == (1, 1024))[0]
        self.exaggeration = exaggeration

    def _prompt(self, text, language, reference, guidance):
        ids = np.asarray(self.tokenizer.encode(text, language))
        if len(ids) > len(self.text_positions):
            raise ValueError(f'The line is too long: {len(ids)} tokens, at most {len(self.text_positions)}.')
        start = self.speech_embedding[START_SPEECH] + self.speech_positions[0]
        rows = []
        for unconditional in ([False, True] if guidance else [False]):
            # The unconditional half keeps the text's positions, not its
            # tokens: upstream zeroes the text embeddings, then adds them.
            text_rows = self.text_positions[np.arange(len(ids))] + (0 if unconditional else self.text_embedding[ids])
            rows.append(np.concatenate([reference.conditioning[0], (self.exaggeration * self.emotion)[None],
                                        text_rows, start[None], start[None]]))
        return np.stack(rows).astype(np.float32)

    def _embed_speech(self, token, step):
        return (self.speech_embedding[token] + self.speech_positions[step])[None, None].astype(np.float32)


class Turbo(_Chatterbox):
    """ResembleAI/chatterbox-turbo-ONNX: English, with paralinguistic tags."""

    name = 'turbo'
    attempts = (Attempt(0.0, False), Attempt(0.0, True), Attempt(0.0, True))
    sampling = {'temperature': 0.8, 'top_k': 1000, 'top_p': 0.95, 'min_p': 0.0}
    normalize_reference = True

    def __init__(self, files: dict[str, Path], threads: int):
        super().__init__(files, threads)
        self.tokenizer = TurboTokenizer(files['tokenizer'])
        self.embed = _session(files['embed_tokens'], threads)

    def _prompt(self, text, language, reference, guidance):
        ids = np.asarray([self.tokenizer.encode(text)], dtype=np.int64)
        text_rows = self.embed.run(None, {'input_ids': ids})[0]
        return np.concatenate([reference.conditioning, text_rows], axis=1).astype(np.float32)

    def _embed_speech(self, token, step):
        return self.embed.run(None, {'input_ids': np.asarray([[token]], dtype=np.int64)})[0]


def load(model: str, files: dict[str, Path], threads: int, exaggeration: float = 0.5) -> _Chatterbox:
    if model == 'turbo':
        return Turbo(files, threads)
    return Multilingual(files, threads, exaggeration=exaggeration)


def graph_files(model: str, local_path: Callable[[str, str], Path]) -> dict[str, Path]:
    """{role: path} for a model, from FILES and a (repo key, path) → file map."""
    out = {}
    for repo, path in FILES[model]:
        if path.endswith('.onnx_data'):
            continue
        name = Path(path).stem
        role = 'tokenizer' if name == 'tokenizer' else name.removesuffix('_q4')
        out[role] = local_path(repo, path)
    return out
