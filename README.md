# Chatterbox add-on for Subtitld

Offline **voice-cloning text-to-speech** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by Resemble AI's
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT), run on
[ONNX Runtime](https://onnxruntime.ai).

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **Clones a voice from a few seconds of speech.** Subtitld sends a clip of
  the speaker from the video, and the dub speaks in their voice, in any of
  the add-on's languages, whatever language the clip is in. Preset voices are
  there for lines with no clip, or to pick a different voice.
- **Two models.** *Chatterbox Multilingual* (0.5B, the default) speaks 20
  languages. *Chatterbox Turbo* (350M) speaks English only, sounds closer to
  the speaker, needs less memory and reads tags such as `[laugh]`, `[sigh]`
  and `[cough]` in the text.
- **CPU only.** About real time on a laptop CPU, with about 1.7 GB of memory
  (Turbo 1 GB). No GPU and no torch.
- **Offline after the first line.** A model downloads on its first line
  (Multilingual 1.3 GB, Turbo 0.7 GB), then nothing leaves the machine: no
  telemetry, no account.

## Languages

Multilingual speaks Arabic, Danish, Dutch, English, Finnish, French, German,
Greek, Hindi, Italian, Korean, Malay, Norwegian, Polish, Portuguese, Russian,
Spanish, Swahili, Swedish and Turkish. With Turbo picked, lines in a language
other than English are spoken by Multilingual.

Chatterbox Multilingual also has Hebrew, Japanese and Chinese, which the
add-on does not offer: upstream adds Hebrew's vowel points with a separate
model and turns Japanese kanji into kana with pykakasi (GPL), and Chinese
came out garbled in our test of the ONNX export.

## Voices

| Voice | What it clones |
|-------|----------------|
| Speaker's own voice | The clip of the speaker Subtitld sends with each line, or the fallback clip in Configure |
| English woman, English man | A LibriVox reader of an English book |
| Portuguese woman, Portuguese man | A LibriVox reader of a Portuguese book |
| French woman, French man | A LibriVox reader of a French book |
| German woman, German man | A LibriVox reader of a German book |

A preset speaks every language; its own language only gives it an accent, so
pick one from the project's language for the most natural dub. The presets
are LibriVox audiobook readers, whose recordings are in the public domain
(the same clips as Subtitld's Sopro add-on); they are credited in
[chatterbox_addon/presets/README.md](chatterbox_addon/presets/README.md).

A good clip to clone is 5 to 20 seconds of one person speaking, with no music
or other voices. The add-on uses the first 10 seconds, cut at a pause, and
repeats a clip shorter than 6 seconds: Turbo babbles with less than 5.

## Options

Set these in **Settings ▸ Add-ons ▸ Chatterbox ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| Model | Multilingual | `CHATTERBOX_MODEL` | `multilingual` or `turbo`. One model is in memory at a time. |
| Expressiveness | 0.5 | `CHATTERBOX_EXPRESSIVENESS` | Multilingual's "exaggeration", 0.25 to 2. Higher is more dramatic, and faster. |
| CPU threads | automatic | `CHATTERBOX_THREADS` | 0 picks half the logical processors, at most 8. |
| Fallback reference clip | none | `CHATTERBOX_VOICE_REF_AUDIO` | Cloned when Subtitld has no clip of the speaker to send. |

Subtitld stretches a dub to fit its subtitle itself, so the add-on ignores
the rate.

## How it runs

Both models come as ONNX exports, split into four graphs: a speech encoder
for the reference clip, token embeddings, the language model with its KV
cache, and the decoder from speech tokens to the waveform. The add-on runs
the loop between them, a port of the exports' model-card examples, plus what
upstream's own pipeline does around the model:

- `chatterbox_addon/text.py`: upstream's punctuation cleanup, lowercasing
  and Unicode decomposition (Multilingual's vocabulary spells "ś" as "s" plus
  an accent), Korean jamo, and the two BPE tokenizers, read from the models'
  tokenizer.json. `tests/test_text.py` checks them against Hugging Face's
  `tokenizers`.
- `chatterbox_addon/dsp.py`: the reference clip's resampling, length, and
  loudness (BS.1770, -27 LUFS for Turbo, as upstream).
- `chatterbox_addon/engine.py`: the graphs and the decoding loop. Both models
  use the 4-bit language models (the 8-bit ones are about seven times slower
  on a CPU) and Turbo's two-step decoder, which decodes Multilingual's speech
  tokens too, about four times faster than its own ten-step one.

Upstream samples and, for Multilingual, uses classifier-free guidance, with
a monitor of the model's attention that stops a line that runs on. The
exports give no attention maps, and without the monitor sampling ran on in a
quarter of our test lines, so the add-on decodes greedily, with upstream's
repetition penalty. When a line gets stuck repeating one sound, it is tried
again with guidance (Multilingual) or sampling (Turbo).

The add-on uses the Multilingual export by onnx-community, which is
Chatterbox Multilingual's first checkpoint (`t3_23lang`); upstream's later
`v2` and `v3` checkpoints have no ONNX export yet.

Measured on an Intel i7-13700T (8 threads), with 10 sentences in English,
Portuguese, French, German, Spanish, Italian, Russian and Polish, each cloned
from two voices and scored with Whisper and Resemblyzer: Multilingual had a
word error rate of 0.024 and a speaker similarity of 0.78; Turbo, on the two
English sentences, 0 and 0.83. Both ran at about real time. The other
languages were checked only with Whisper's small model, which hears several
of them poorly, and none of them by ear.

## What is bundled, what is downloaded

| What | Size | Licence | Source |
|------|------|---------|--------|
| Add-on, ONNX Runtime, numpy (bundled) | see the release | Apache-2.0; MIT; BSD-3-Clause | this repository, PyPI |
| Preset voice clips (bundled) | about 4 MB | public domain; CC-BY-4.0 | LibriVox, LibriTTS-R |
| Multilingual (downloaded) | 1,015 MB + Turbo's decoder, 249 MB | MIT | [onnx-community/chatterbox-multilingual-ONNX](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX) |
| Turbo (downloaded) | 725 MB | MIT | [ResembleAI/chatterbox-turbo-ONNX](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) |

The models go to Subtitld's model folder (`$SUBTITLD_MODELS_DIR/chatterbox`),
at the revisions in `chatterbox_addon/pins.json`, each file checked by its
sha256. An interrupted download resumes. `$HF_ENDPOINT` points it at a
mirror.

Each release archive carries `THIRD_PARTY_LICENSES/README.md`, which names
the notice for every native file in it; the build fails if one has none. The
Linux archive carries libquadmath (LGPL-2.1), from numpy's wheel; its source
is attached to the same release.

Upstream marks its own audio with Resemble AI's Perth watermark. This add-on
does not (Perth needs torch). Clone a voice only with its owner's consent.

## Limits

- **Twenty languages.** A project in any other language is refused with a
  message that says so.
- **Not the same samples twice.** The same line in the same voice gives the
  same words, timing and voice every time, but the decoder starts from noise
  it draws itself, so the waveform differs slightly between runs.
- **Long lines.** A line is spoken in one go, up to 40 seconds of speech.
- **Platforms.** Releases ship Linux x86_64, macOS arm64 and Windows x86_64.
- **One line at a time.** A cancel stops the line between model steps, or
  stops the download, which resumes on the next line.

## Build

```bash
pip install -c constraints.txt -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/chatterbox-addon/chatterbox-addon --self-test
CHATTERBOX_TEST_BINARY=$PWD/dist/chatterbox-addon/chatterbox-addon pytest tests/test_installed.py
```

To move to other revisions of the exports, run `tools/pin_model.py` and
check that the tests still pass.

A tag `vYY.MM.N` builds, tests and publishes every platform
(`.github/workflows/release.yml`).
