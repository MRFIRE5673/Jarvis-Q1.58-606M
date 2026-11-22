"""Prepare a streamed code-and-conversation instruction-tuning corpus."""
import argparse
import hashlib
import os
import time

import fsspec
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from tokenizer import ENCODING, EOT


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "Nan-Do/instructional_code-search-net-python"
LICENSE = "Apache-2.0"
CHAT_REPO = "OpenAssistant/oasst1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=20_000_000)
    parser.add_argument("--chat-tokens", type=int, default=5_000_000,
                        help="token budget for general English conversation examples")
    parser.add_argument("--max-example-tokens", type=int, default=512)
    parser.add_argument("--output", default=os.path.join("data", "assistant_sft_tokens.npy"))
    parser.add_argument("--mask-only", action="store_true",
                        help="build the response-loss mask from an existing token file without downloading data")
    args = parser.parse_args()
    if args.tokens < 8192 or args.chat_tokens < 0 or args.tokens - args.chat_tokens < 8192:
        parser.error("--tokens must leave at least 8192 code tokens, and --chat-tokens cannot be negative")
    if args.max_example_tokens < 16:
        parser.error("--max-example-tokens must be at least 16")
    return args


def write_target_mask(token_path, mask_path):
    tokens = np.load(token_path, mmap_mode="r")
    if tokens.ndim != 1 or tokens.dtype != np.uint16:
        raise RuntimeError("SFT tokens must be a one-dimensional uint16 GPT-2 token array")

    marker = np.asarray(ENCODING.encode_ordinary("\nAssistant: "), dtype=tokens.dtype)
    eot_positions = np.flatnonzero(tokens == EOT)
    mask = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.uint8, shape=tokens.shape
    )
    mask[:] = 0
    start = 0
    examples = 0
    active_tokens = 0
    for end in eot_positions:
        end = int(end)
        segment = tokens[start:end]
        candidates = np.flatnonzero(segment[:max(0, segment.size - len(marker) + 1)] == marker[0])
        matches = [
            int(offset) for offset in candidates
            if np.array_equal(segment[offset:offset + len(marker)], marker)
        ]
        if not matches:
            raise RuntimeError(
                f"Expected an assistant delimiter in SFT example {examples + 1}, found none"
            )
        response_start = start + matches[-1] + len(marker)
        mask[response_start:end + 1] = 1
        active_tokens += end + 1 - response_start
        examples += 1
        start = end + 1

    if start < tokens.size:
        tail = tokens[start:]
        candidates = np.flatnonzero(tail[:max(0, tail.size - len(marker) + 1)] == marker[0])
        matches = [
            int(offset) for offset in candidates
            if np.array_equal(tail[offset:offset + len(marker)], marker)
        ]
        if matches:
            response_start = start + matches[-1] + len(marker)
            mask[response_start:tokens.size] = 1
            active_tokens += tokens.size - response_start
    if examples == 0 or active_tokens == 0:
        raise RuntimeError("SFT token file has no complete response targets")
    mask.flush()
    del mask
    return examples, active_tokens


def update_metadata(metadata_path, mask_name):
    existing = []
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            existing = [
                line.rstrip("\n") for line in f
                if not line.startswith(("target_mask=", "loss_mask="))
            ]
    existing.extend((f"target_mask={mask_name}", "loss_mask=response-only-v1"))
    temp = metadata_path + ".tmp"
    with open(temp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(existing) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, metadata_path)


def iter_rows():
    api = HfApi()
    paths = sorted(
        path for path in api.list_repo_files(REPO, repo_type="dataset")
        if path.endswith(".parquet")
    )
    if not paths:
        raise RuntimeError(f"No Parquet shards found in {REPO}")
    for index, path in enumerate(paths, 1):
        print(f"[sft-data] Reading shard {index}/{len(paths)}: {path}", flush=True)
        url = f"hf://datasets/{REPO}/{path}"
        with fsspec.open(url, "rb", block_size=1 << 20) as source:
            parquet = pq.ParquetFile(source)
            required = {"INSTRUCTION", "RESPONSE"}
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise RuntimeError(f"{REPO}/{path} is missing columns: {sorted(missing)}")
            for batch in parquet.iter_batches(
                batch_size=128, columns=["INSTRUCTION", "RESPONSE"], use_threads=False
            ):
                yield from batch.to_pylist()


def iter_chat_rows():
    """Yield reviewed English root-prompt/reply pairs from OASST1 train."""
    repo = CHAT_REPO
    paths = sorted(
        path for path in HfApi().list_repo_files(repo, repo_type="dataset")
        if path.startswith("data/train-") and path.endswith(".parquet")
    )
    if not paths:
        raise RuntimeError(f"No training Parquet shards found in {repo}")

    messages = {}
    for index, path in enumerate(paths, 1):
        print(f"[sft-data] Reading chat shard {index}/{len(paths)}: {path}", flush=True)
        with fsspec.open(f"hf://datasets/{repo}/{path}", "rb", block_size=1 << 20) as source:
            parquet = pq.ParquetFile(source)
            required = {"message_id", "parent_id", "text", "role", "lang"}
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise RuntimeError(f"{repo}/{path} is missing columns: {sorted(missing)}")
            optional = {"deleted", "review_result", "synthetic"}
            columns = sorted(required | (optional & set(parquet.schema_arrow.names)))
            for batch in parquet.iter_batches(batch_size=256, columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    if row.get("lang") != "en" or row.get("deleted") or row.get("review_result") is False:
                        continue
                    if row.get("synthetic") is True:
                        continue
                    text = row.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    message_id = row.get("message_id")
                    parent_id = row.get("parent_id")
                    role = row.get("role")
                    messages[message_id] = (role, text.strip(), parent_id)
                    if role == "assistant" and parent_id in messages:
                        turns = []
                        current_id = parent_id
                        while current_id in messages:
                            parent_role, parent_text, grandparent_id = messages[current_id]
                            turns.append(("User" if parent_role == "prompter" else "Assistant", parent_text))
                            current_id = grandparent_id
                        turns.reverse()
                        if turns and turns[-1][0] == "User":
                            prompt = "\n".join(f"{speaker}: {body}" for speaker, body in turns) + "\nAssistant: "
                            yield {"PROMPT": prompt, "INSTRUCTION": prompt, "RESPONSE": text.strip()}


def encode_example(row, max_tokens):
    instruction, response = row.get("INSTRUCTION"), row.get("RESPONSE")
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    if not isinstance(response, str) or not response.strip():
        return None

    prompt = row.get("PROMPT")
    if not isinstance(prompt, str):
        prompt = f"User: {instruction.strip()}\nAssistant: "
    prompt_ids = ENCODING.encode_ordinary(prompt)
    response_ids = ENCODING.encode_ordinary(response.strip() + "\n")
    content_budget = max_tokens - 1  # reserve one GPT-2 end-of-text token
    prompt_budget = content_budget // 2
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    response_ids = response_ids[:content_budget - len(prompt_ids)]
    if not response_ids:
        return None
    return prompt_ids + response_ids + [EOT], len(prompt_ids)


def trim_npy(path, size):
    source = np.load(path, mmap_mode="r")
    trimmed_path = path + ".trim.npy"
    try:
        with open(trimmed_path, "wb") as output:
            np.save(output, source[:size], allow_pickle=False)
        source._mmap.close()
        source = None
        os.replace(trimmed_path, path)
    finally:
        if source is not None:
            source._mmap.close()
        if os.path.exists(trimmed_path):
            os.remove(trimmed_path)


def is_holdout(row):
    key = f"{row['INSTRUCTION']}\0{row['RESPONSE']}".encode("utf-8", errors="replace")
    return hashlib.sha256(key).digest()[0] < 13  # deterministic 5% split


def main():
    args = arguments()
    output = os.path.abspath(os.path.join(ROOT, args.output))
    holdout = os.path.splitext(output)[0] + "_holdout.txt"
    mask_output = os.path.splitext(output)[0] + "_mask.npy"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temp_output, temp_holdout = output + ".tmp.npy", holdout + ".tmp"
    temp_mask = mask_output + ".tmp.npy"

    if args.mask_only:
        if not os.path.exists(output):
            raise FileNotFoundError(f"Cannot build SFT mask; token file is missing: {output}")
        try:
            examples, active_tokens = write_target_mask(output, temp_mask)
            os.replace(temp_mask, mask_output)
            update_metadata(os.path.splitext(output)[0] + ".meta", os.path.basename(mask_output))
            print(f"[sft-data] Response-only mask: {active_tokens:,} targets across {examples:,} examples -> {mask_output}")
        finally:
            if os.path.exists(temp_mask):
                os.remove(temp_mask)
        return
    code_tokens = args.tokens - args.chat_tokens
    mapped = mask_mapped = None
    rows_seen = train_examples = holdout_examples = holdout_tokens = pos = 0
    code_train_examples = code_holdout_examples = chat_train_examples = chat_holdout_examples = 0
    actual_code_tokens = actual_chat_tokens = 0
    next_report = 1_000_000
    started = time.monotonic()

    try:
        mapped = np.lib.format.open_memmap(
            temp_output, mode="w+", dtype=np.uint16, shape=(args.tokens,)
        )
        mask_mapped = np.lib.format.open_memmap(
            temp_mask, mode="w+", dtype=np.uint8, shape=(args.tokens,)
        )
        mask_mapped[:] = 0
        with open(temp_holdout, "w", encoding="utf-8", newline="\n") as holdout_file:
            code_cache = os.path.join(ROOT, "data", "code_instruction_sft_tokens.npy")
            code_holdout = os.path.join(ROOT, "data", "code_instruction_sft_tokens_holdout.txt")
            code_meta = os.path.join(ROOT, "data", "code_instruction_sft_tokens.meta")
            code_mask = os.path.join(ROOT, "data", "code_instruction_sft_tokens_mask.npy")
            reused_code = False
            if (os.path.abspath(code_cache) != output and os.path.exists(code_cache)
                    and os.path.exists(code_mask)):
                cached = np.load(code_cache, mmap_mode="r")
                cached_mask = np.load(code_mask, mmap_mode="r")
                if (cached.ndim == 1 and cached.dtype == np.uint16 and cached.size >= code_tokens
                        and cached_mask.ndim == 1 and cached_mask.dtype == np.uint8
                        and cached_mask.shape == cached.shape):
                    ends = np.flatnonzero(cached[:code_tokens] == EOT)
                    if not ends.size:
                        raise RuntimeError("Local code SFT cache has no complete examples")
                    pos = int(ends[-1]) + 1
                    actual_code_tokens = pos
                    mapped[:pos] = cached[:pos]
                    mask_mapped[:pos] = cached_mask[:pos]
                    code_train_examples = int(ends.size)
                    if os.path.exists(code_holdout):
                        with open(code_holdout, encoding="utf-8", errors="replace") as f:
                            holdout_file.write(f.read())
                    if os.path.exists(code_meta):
                        with open(code_meta, encoding="utf-8") as f:
                            meta = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
                        code_holdout_examples = int(meta.get("holdout_examples", 0))
                        code_holdout_tokens = int(meta.get("holdout_tokens", 0))
                        holdout_examples += code_holdout_examples
                        holdout_tokens += code_holdout_tokens
                    train_examples += code_train_examples
                    reused_code = True
                    print(f"[sft-data] Reused {actual_code_tokens:,} complete local code tokens", flush=True)
                cached_mask._mmap.close()
                del cached
                del cached_mask

            def write_rows(rows, token_limit, label):
                nonlocal pos, rows_seen, train_examples, holdout_examples, holdout_tokens, next_report
                nonlocal code_train_examples, code_holdout_examples, chat_train_examples, chat_holdout_examples
                source_tokens = 0
                for row in rows:
                    rows_seen += 1
                    encoded = encode_example(row, args.max_example_tokens)
                    if encoded is None:
                        continue
                    ids, target_start = encoded
                    if is_holdout(row):
                        holdout_file.write(ENCODING.decode(ids[:-1]) + "\n\n")
                        holdout_examples += 1
                        holdout_tokens += len(ids)
                        if label == "chat":
                            chat_holdout_examples += 1
                        else:
                            code_holdout_examples += 1
                        continue

                    count = len(ids)
                    if count > token_limit - source_tokens:
                        return
                    mapped[pos:pos + count] = ids
                    mask_mapped[pos + target_start:pos + count] = 1
                    pos += count
                    source_tokens += count
                    train_examples += 1
                    if label == "chat":
                        chat_train_examples += 1
                    else:
                        code_train_examples += 1
                    if pos >= next_report or source_tokens == token_limit:
                        elapsed = time.monotonic() - started
                        print(
                            f"[sft-data] {pos:,}/{args.tokens:,} train tokens; "
                            f"{rows_seen:,} rows; {elapsed:.0f}s",
                            flush=True,
                        )
                        next_report = ((pos // 1_000_000) + 1) * 1_000_000
                    if source_tokens == token_limit:
                        return
                raise RuntimeError(f"{label} source ended after {source_tokens:,}/{token_limit:,} training tokens")

            if not reused_code:
                write_rows(iter_rows(), code_tokens, "code")
                actual_code_tokens = pos
            if args.chat_tokens:
                write_rows(iter_chat_rows(), args.chat_tokens, "chat")
            actual_chat_tokens = pos - actual_code_tokens

        if pos < args.tokens - (2 * args.max_example_tokens):
            raise RuntimeError(f"Sources yielded only {pos:,}/{args.tokens:,} training tokens")
        if holdout_tokens < 51_201:
            raise RuntimeError(f"Holdout is too small for evaluation: {holdout_tokens:,} tokens")

        mapped.flush()
        mask_mapped.flush()
        del mapped
        del mask_mapped
        mapped = None
        mask_mapped = None
        trim_npy(temp_output, pos)
        trim_npy(temp_mask, pos)
        active_tokens = int(np.count_nonzero(np.load(temp_mask, mmap_mode="r")))
        os.replace(temp_output, output)
        os.replace(temp_mask, mask_output)
        os.replace(temp_holdout, holdout)
        metadata = os.path.splitext(output)[0] + ".meta"
        with open(metadata + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            f.write(
                f"format=user-assistant-mix-v3\ncode_source={REPO}\nchat_source={CHAT_REPO}:train\nlicense={LICENSE}\n"
                f"tokenizer=gpt2\nrequested_tokens={args.tokens}\nrequested_code_tokens={code_tokens}\n"
                f"requested_chat_tokens={args.chat_tokens}\ntokens={pos}\ncode_tokens={actual_code_tokens}\n"
                f"chat_tokens={actual_chat_tokens}\nactive_response_tokens={active_tokens}\n"
                f"max_example_tokens={args.max_example_tokens}\n"
                f"train_examples={train_examples}\nholdout_examples={holdout_examples}\n"
                f"code_train_examples={code_train_examples}\ncode_holdout_examples={code_holdout_examples}\n"
                f"chat_train_examples={chat_train_examples}\nchat_holdout_examples={chat_holdout_examples}\n"
                f"holdout_tokens={holdout_tokens}\nholdout_file={os.path.basename(holdout)}\n"
                f"target_mask={os.path.basename(mask_output)}\n"
                f"loss_mask=response-only-v1\n"
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(metadata + ".tmp", metadata)
        print(
            f"[sft-data] Ready: {actual_code_tokens:,} code + {actual_chat_tokens:,} chat train tokens "
            f"({active_tokens:,} response targets), "
            f"{holdout_tokens:,} held-out tokens -> {output}",
            flush=True,
        )
    finally:
        if mapped is not None:
            del mapped
        if mask_mapped is not None:
            del mask_mapped
        for path in (temp_output, temp_holdout, temp_mask,
                     temp_output + ".trim.npy", temp_mask + ".trim.npy"):
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()
