import os
import time
import math
import tiktoken
import torch
from model import GPT, GPTConfig
from data_loader import DataLoaderLite
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA needed for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank==0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
      device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backend.mps.is_available():
      device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)


total_batch_size = 131072
B = 4
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
  print(f"total desired batch size: {total_batch_size}")
  print(f"calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'train', master_process)
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, 'val', master_process)


torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 #715 for new data
max_steps = 19073 #19073 for new data

def get_lr(it):
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  if it > max_steps:
    return min_lr

  decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate=6e-4, device_type=device_type)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
  pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step%250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
          val_loss_accum = 0.0
          val_loss_steps = 20
          for _ in range(val_loss_steps):
              x, y = val_loader.next_batch()
              x, y = x.to(device), y.to(device)
              with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
              loss = loss / val_loss_steps
              val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    if (step>0 and step%250==0) or last_step:
      model.eval()
      num_of_return_seq = 4
      max_length = 32
      enc = tiktoken.get_encoding('gpt2')
      tokens = enc.encode('India is')
      tokens = torch.tensor(tokens, dtype=torch.long)
      tokens = tokens.unsqueeze(0).repeat(num_of_return_seq, 1)
      xgen = tokens.to(device)

      sample_rng = torch.Generator(device=device)
      sample_rng.manual_seed(42+ddp_rank)
      while x.size(1) < max_length:
          with torch.no_grad():
              with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen)
              logits = logits[:, -1, :]
              probs = F.softmax(logits, dim=-1)
              topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
              ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
              xcol = torch.gather(topk_indices, -1, ix)
              xgen = torch.cat((xgen, xcol), dim=1)

      for i in range(num_of_return_seq):
          tokens = xgen[i, :max_length].tolist()
          decoded = enc.decode(tokens)
          print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      if ddp:
          model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, loss = model(x, y)
      loss = loss / grad_accum_steps
      loss_accum += loss.detach()
      loss.backward()
    if ddp:
      dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
      torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tps = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1-t0)
    if master_process:
      print(f'step {step:4d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tps:.2f}')
      with open(log_file, 'a') as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
  destroy_process_group()