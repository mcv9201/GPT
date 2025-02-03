import torch
import tiktoken
from model import GPT, GPTConfig
from torch.nn import functional as F

class GPTPredict():

    def __init__(self, checkpoint_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint['model'], strict=False)

    def generate(self, input, length=32):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        num_of_return_seq = 1
        max_length = length
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(input)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_of_return_seq, 1)
        xgen = tokens.to(device)
        prev = 0
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = self.model(xgen, None)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

                tokens = xgen[0, prev:xgen.size(1)].tolist()
                prev = xgen.size(1)
                decoded = enc.decode(tokens)

        tokens = xgen[0, :max_length].tolist()
        decoded = enc.decode(tokens)
        return decoded