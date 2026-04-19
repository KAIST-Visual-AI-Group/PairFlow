import einops


class DummyVisionTokenizer:
  def __init__(self, vocab_size, image_size,
               add_mask_token=True,
               add_special_tokens=True):
    self.pad_token_id = None
    self.pad_token = None
    if add_mask_token:
      self.mask_token = vocab_size
      self.mask_token_id = vocab_size
      self.vocab_size = vocab_size + 1
    else:
      self.vocab_size = vocab_size
    if add_special_tokens:
      self.bos_token_id = vocab_size
      self.bos_token = vocab_size
      self.eos_token_id = vocab_size + 1
      self.eos_token = vocab_size + 1
      self.vocab_size = self.vocab_size + 2
    else:
      self.bos_token = None
      self.eos_token = None
      self.vocab_size = self.vocab_size
    self.image_size = image_size

  def __call__(self, x):
    return x

  def __len__(self):
    return self.vocab_size

  def batch_decode(self, x):
    return einops.rearrange(x, "b (c h w) -> b c h w", c=3,
                     h=self.image_size)

  def decode(self, x):
    return einops.rearrange(x, "(c h w) -> c h w", c=3,
                     h=self.image_size)
