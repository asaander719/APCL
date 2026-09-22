"""Build the standalone example using an explicit source/document allowlist."""
import argparse
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = [root/'README.md', root/'reproduction/README.md',
             root/'reproduction/requirements.txt', root/'reproduction/package.py']
    for folder, pattern in [('apcl_repro','*.py'), ('reproduction/configs','*.json'),
                            ('reproduction/example','*.json'), ('reproduction/example','*.jsonl'),
                            ('reproduction/tests','*.py')]:
        files.extend(sorted((root/folder).rglob(pattern)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            archive.write(file, 'APCL-reproduction/'+str(file.relative_to(root)))
    print(a.output.resolve(), a.output.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
