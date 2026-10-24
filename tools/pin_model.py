"""Write chatterbox_addon/pins.json: every file the two models read, pinned.

    python tools/pin_model.py [MULTILINGUAL_REVISION [TURBO_REVISION]]

Without revisions, pins each repository's current head. Sizes and sha256 come
from Hugging Face's LFS metadata; the small files that are not in LFS are
downloaded and hashed.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from chatterbox_addon.engine import FILES  # noqa: E402

REPOS = {
    'multilingual': 'onnx-community/chatterbox-multilingual-ONNX',
    'turbo': 'ResembleAI/chatterbox-turbo-ONNX',
}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> int:
    wanted = sorted({entry for files in FILES.values() for entry in files})
    out = {}
    for number, (key, repo) in enumerate(REPOS.items()):
        revision = sys.argv[1 + number] if len(sys.argv) > 1 + number else \
            json.loads(get(f'https://huggingface.co/api/models/{repo}'))['sha']
        tree = {item['path']: item for item in json.loads(get(
            f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1'))}
        files = []
        for _, path in (entry for entry in wanted if entry[0] == key):
            meta = tree[path]
            lfs = meta.get('lfs') or {}
            sha = lfs.get('oid') or lfs.get('sha256')
            if not sha:
                data = get(f'https://huggingface.co/{repo}/resolve/{revision}/{path}')
                if len(data) != meta['size']:
                    raise SystemExit(f'{repo}/{path}: got {len(data)} bytes, the tree says {meta["size"]}')
                sha = hashlib.sha256(data).hexdigest()
            files.append({'path': path, 'size': meta['size'], 'sha256': sha})
        out[key] = {'repo': repo, 'revision': revision, 'files': files}
        print(f"{repo}: {len(files)} files, {sum(f['size'] for f in files) / 1e6:.0f} MB at {revision}")
    (HERE / 'chatterbox_addon' / 'pins.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
