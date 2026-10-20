"""Stream GPT-2 tokens into a disk-backed English/code training corpus."""
import argparse
import os
import time

import fsspec
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tokenizer import ENCODING, EOT


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERMISSIVE_LICENSES = {
    "mit", "isc", "bsd-2-clause", "bsd-3-clause", "apache-2.0",
    "unlicense", "zlib", "cc0-1.0",
}
LICENSE_LABEL = "MIT,ISC,BSD-2-Clause,BSD-3-Clause,Apache-2.0,Unlicense,Zlib,CC0-1.0"
BLOCK = 8192  # one CUDA optimizer step at the default B=8, T=512, ACC=2
def list_parquet_shards(repo, prefix):
    print(f"Listing Parquet shards for {repo}...", flush=True)
    paths = [
        path for path in HfApi().list_repo_files(repo, repo_type="dataset")
        if path.startswith(prefix) and path.endswith(".parquet")
    ]
    paths.sort()
    if not paths:
        raise RuntimeError(f"No Parquet shards found in {repo}/{prefix}")
    print(f"  found {len(paths):,} shards; reads are limited to what's needed", flush=True)
    return paths


def iter_parquet_rows(repo, paths, label, columns):
    for index, path in enumerate(paths, 1):
        print(f"  {label}: reading shard {index:,}/{len(paths):,}: {path}", flush=True)
        url = f"hf://datasets/{repo}/{path}"
        # Arrow reads only requested columns and Parquet row groups over ranged
        # requests. Unlike load_dataset's wildcard resolver, this never indexes
        # or downloads every shard before yielding its first row.
        with fsspec.open(url, "rb", block_size=1 << 20) as source:
            parquet = pq.ParquetFile(source)
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise RuntimeError(f"{repo}/{path} lacks expected columns: {sorted(missing)}")
            for batch in parquet.iter_batches(batch_size=32, columns=columns, use_threads=False):
                yield from batch.to_pylist()


def token_chunks(dataset, field, label, filter_code=False):
    reported = 0
    total = 0
    rows = 0
    last_report = time.monotonic()
    for row in dataset:
        rows += 1
        if filter_code and str(row.get("license", "")).strip().lower() not in PERMISSIVE_LICENSES:
            if rows % 1000 == 0 and time.monotonic() - last_report >= 5:
                print(f"  {label}: scanned {rows:,} rows, accepted {total:,} tokens", flush=True)
                last_report = time.monotonic()
            continue
        text = row.get(field)
        if not isinstance(text, str) or not text:
            continue
        ids = ENCODING.encode_ordinary(text)
        if ids:
            ids.append(EOT)
            total += len(ids)
            if total >= reported + 1_000_000:
                print(f"  {label}: {total:,} tokens", flush=True)
                reported = total
                last_report = time.monotonic()
            yield np.asarray(ids, dtype=np.uint16)


class TokenStream:
    def __init__(self, chunks, label):
        self.chunks = iter(chunks)
        self.label = label
        self.buffer = np.empty(0, dtype=np.uint16)
        self.offset = 0

    def read(self, count):
        out = np.empty(count, dtype=np.uint16)
        written = 0
        while written < count:
            if self.offset == self.buffer.size:
                try:
                    self.buffer = next(self.chunks)
                except StopIteration as exc:
                    raise RuntimeError(f"{self.label} source ended after {written:,} tokens of a requested block") from exc
                self.offset = 0
            take = min(count - written, self.buffer.size - self.offset)
            out[written:written + take] = self.buffer[self.offset:self.offset + take]
            self.offset += take
            written += take
        return out

    def close(self):
        close = getattr(self.chunks, "close", None)
        if close:
            close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", nargs="?", type=int, help="legacy English-only token count (default: 20M)")
    parser.add_argument("--english-tokens", type=int)
    parser.add_argument("--code-tokens", type=int)
    parser.add_argument("--output", default=os.path.join("data", "train_tokens.npy"))
    args = parser.parse_args()
    if args.english_tokens is None and args.code_tokens is None:
        args.english_tokens = 20_000_000 if args.tokens is None else args.tokens
        args.code_tokens = 0
    elif args.tokens is not None or args.english_tokens is None or args.code_tokens is None:
        parser.error("use either [tokens] or both --english-tokens and --code-tokens")
    if args.english_tokens < 0 or args.code_tokens < 0 or args.english_tokens + args.code_tokens <= 0:
        parser.error("token counts must be non-negative and their sum must be positive")
    return args


def main():
    args = arguments()
    english_n, code_n = args.english_tokens, args.code_tokens
    total = english_n + code_n
    output = os.path.abspath(os.path.join(ROOT, args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temp = output + ".tmp.npy"

    english = None
    code = None
    mapped = None
    try:
        if english_n:
            repo = "HuggingFaceFW/fineweb-edu"
            paths = list_parquet_shards(repo, "sample/10BT/")
            rows = iter_parquet_rows(repo, paths, "FineWeb-Edu", ["text"])
            english = TokenStream(token_chunks(rows, "text", "FineWeb-Edu"), "FineWeb-Edu")
        if code_n:
            # Rows use the `content` field. Reading `code` silently skipped every row.
            repo = "codeparrot/github-code"
            paths = list_parquet_shards(repo, "data/")
            paths = [path for path in paths if "/train-" in path]
            rows = iter_parquet_rows(repo, paths, "GitHub Code", ["content", "license"])
            code = TokenStream(token_chunks(rows, "content", "GitHub Code", filter_code=True), "licensed GitHub Code")

        mapped = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint16, shape=(total,))
        pos = 0
        next_report = 5_000_000
        while pos < total:
            count = min(BLOCK, total - pos)
            next_pos = pos + count
            en_count = (next_pos * english_n // total) - (pos * english_n // total)
            if en_count:
                mapped[pos:pos + en_count] = english.read(en_count)
            code_count = count - en_count
            if code_count:
                mapped[pos + en_count:next_pos] = code.read(code_count)
            pos = next_pos
            if pos >= next_report or pos == total:
                print(f"  mixed corpus: {pos:,}/{total:,} tokens", flush=True)
                next_report = ((pos // 5_000_000) + 1) * 5_000_000

        mapped.flush()
        del mapped
        mapped = None
        os.replace(temp, output)
        mode = "english-only" if not code_n else "english-code-mix"
        metadata = output[:-4] + ".meta" if output.endswith(".npy") else output + ".meta"
        metadata_tmp = metadata + ".tmp"
        with open(metadata_tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(
                f"mode={mode}\ntokens={total}\nenglish_tokens={english_n}\ncode_tokens={code_n}\n"
                f"english_source=HuggingFaceFW/fineweb-edu:sample-10BT\n"
                f"code_source=codeparrot/github-code:Parquet\n"
                f"code_licenses={LICENSE_LABEL if code_n else 'none'}\n"
            )
        os.replace(metadata_tmp, metadata)
    finally:
        if english is not None:
            english.close()
        if code is not None:
            code.close()
        if mapped is not None:
            del mapped
        if os.path.exists(temp):
            os.remove(temp)
    print(f"saved {total:,} tokens ({english_n:,} English, {code_n:,} code) -> {output}")


if __name__ == "__main__":
    main()
