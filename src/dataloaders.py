import torch, itertools
from pathlib import Path
import glob

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int, cycle: bool = True, seq_len: int = 1024, device: str = "cuda"):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = itertools.cycle(files) if cycle else iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        # Check if we have enough tokens for all batches and sequences
        # CHANGED FOR V2: need tokens for contexts + single target per batch
        # OLD: total_tokens_needed = batch_size * seq_len + 1  # +1 for target shift
        total_tokens_needed = batch_size * seq_len + batch_size  # seq_len tokens per batch + 1 target per batch
        if pos + total_tokens_needed >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except StopIteration:
                if not cycle:
                    break
                file_iter = iter(files)
                tokens, pos = _load_data_shard(next(file_iter)), 0
        
        # Extract tokens for this rank's local batch
        # CHANGED FOR V2: extract context + single target tokens per batch
        # OLD: start_idx = pos + rank * local_batch_size * seq_len  
        # OLD: end_idx = start_idx + local_batch_size * seq_len + 1  # +1 for target shift
        start_idx = pos + rank * (local_batch_size * seq_len + local_batch_size)
        end_idx = start_idx + local_batch_size * seq_len + local_batch_size
        buf = tokens[start_idx:end_idx]
        
        # Reshape into batched sequences
        # CHANGED FOR V2: inputs=[batch_size, seq_len], targets=[batch_size] (single next token)
        # OLD: inputs = buf[:-1].reshape(local_batch_size, seq_len).to(device="cuda", dtype=torch.int32, non_blocking=True)
        # OLD: targets = buf[1:].reshape(local_batch_size, seq_len).to(device="cuda", dtype=torch.int64, non_blocking=True)
        
        # New format: inputs are first seq_len tokens, targets are the next single token
        inputs = buf[:local_batch_size * seq_len].reshape(local_batch_size, seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[local_batch_size * seq_len:local_batch_size * seq_len + local_batch_size].to(device=device, dtype=torch.int64, non_blocking=True)
        
        pos += total_tokens_needed
        yield inputs, targets
