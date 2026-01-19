import os
import time
import h5py
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from data_provider.data_loader_emb import (
    APAVALoader, ADFTDLoader, PTBLoader, PTBXLLoader, TDBRAINLoader, MIMICIVLoader
)
from gen_vlm_emb import GenVLMEmb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="APAVA")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--root_path", type=str, default="./dataset/APAVA/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--divide", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count()))
    parser.add_argument("--amp", action="store_true", help="Enable autocast half-precision for inference")
    parser.add_argument("--skip_existing", action="store_true", help="Skip saving if file exists")
    return parser.parse_args()


def get_dataset(name, root_path, flag):
    datasets = {
        'APAVA': APAVALoader,
        'ADFTD': ADFTDLoader,
        'PTB': PTBLoader,
        'PTB-XL': PTBXLLoader,
        'TDBRAIN': TDBRAINLoader,
        'MIMIC': MIMICIVLoader,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    return datasets[name](root_path=root_path, flag=flag)


def split_indices(n, world_size, rank):
    """Evenly split [0, n) into world_size parts; difference between parts ≤ 1."""
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


def save_shard_embeddings(dataset, split_name, args, device, rank, world_size):
    torch.backends.cudnn.benchmark = True

    total_len = len(dataset)
    shard_indices = split_indices(total_len, world_size, rank)
    shard = Subset(dataset, shard_indices)

    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    model = GenVLMEmb(device=device, fs=args.fs).to(device)
    model.eval()

    save_dir = os.path.join("./emb_VLM", args.data, split_name)
    os.makedirs(save_dir, exist_ok=True)  

    start_time = time.time()
    processed = 0
    global_ptr = 0  
    with torch.inference_mode():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)

            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = model.generate_embeddings(x)
            else:
                emb = model.generate_embeddings(x)

            # emb: [B, ...]
            bs = x.size(0)
            batch_indices = shard_indices[global_ptr: global_ptr + bs]
            global_ptr += bs

            emb_cpu = emb.detach().to("cpu")
            for i_in_batch, global_idx in enumerate(batch_indices):
                file_path = os.path.join(save_dir, f"{global_idx}.h5")
                if args.skip_existing and os.path.exists(file_path):
                    continue
                arr = emb_cpu[i_in_batch].numpy()
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('embeddings', data=arr)

            processed += bs
            if processed % (args.batch_size * 10) == 0 or processed == len(shard):
                print(f"[rank {rank}] {split_name}: processed {processed}/{len(shard)}")

    dur = time.time() - start_time
    print(f"[rank {rank}] Done {split_name} shard: {processed} samples in {dur/60:.2f} min.")


def main_worker():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    splits = ["train", "val", "test"] if args.divide == "all" else [args.divide]

    for split_name in splits:
        dataset = get_dataset(args.data, args.root_path, split_name)
        if rank == 0:
            print(f"[rank 0] Split={split_name}, Total samples={len(dataset)}")

        save_shard_embeddings(dataset, split_name, args, device, rank, world_size)


if __name__ == "__main__":
    main_worker()
