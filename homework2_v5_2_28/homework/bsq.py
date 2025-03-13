import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1 # note: (x >= 0) evals to a bool (1 or 0)
    return x + (sign - x).detach()  
    """
    - so if x was positive, (sign - x) is essentially -x+1 then add +x and get +1
    - if x was neg, (sign-x) is -1+x then the x+ on left side is basically -x so left with -1
    - When you perform an operation that doesn't require gradients, you can use .detach() to ensure 
      that PyTorch doesn't track the gradients of intermediate operations.
      - .detach() is useful if you only want to use that tensor for computation but don't want to backpropagate through it.
    """

# DONT have to write code for this class. Purpose is to define abstract methods (define params)
# that classes that inherit this class will have to define
class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        #raise NotImplementedError() 
        self._codebook_bits=codebook_bits # used by _index_to_code()
        self.linear_down=torch.nn.Linear(embedding_dim, codebook_bits) # A linear down-projection into codebook_bits dimensions
        ####self.l2_norm=torch.nn. -> do NOT use nn.LayerNorm() and do Not use batch norm bc dont want to normalize across batches
        self.linear_up=torch.nn.Linear(codebook_bits, embedding_dim)
        

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization  
            - Notes regarding normalization
                - do not use nn.LayerNorm() nor BatchNorm() bc these normalize across batches
                - after linear down proj, dimension of input is now (batch x height x width x channels) where channels=codebook_bits
                - will want to normalize in the codebook_bits dimension so -1 dimension (last dimension) (want to norm across the tokens on per image basis not across the batch) 
        - differentiable sign
        """
        #raise NotImplementedError()
        return diff_sign(torch.nn.functional.normalize(self.linear_down(x), p=2.0, dim=-1)) # the normalize() funct calcs L_p norm where p=2 by default and we normalize on the codebook_bits dimension (batch x h x w x channel) where channel is codebook_bits size
       

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        #raise NotImplementedError()
        return self.linear_up(x) # notice we're not doing normalization then linear up proj as done in paper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)) # self.encode() and self.decode() calls the def encode() and def decode() in this class

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1



class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim) # call PatchAutoEncoder()'s init() funct which creates self.encoder and self.decoder
        #raise NotImplementedError()
        self.bsq=BSQ(codebook_bits, latent_dim)
        

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
        return self.bsq.encode_index(super().encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
        return super().decode(self.bsq.decode_index(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
        return super().encode(x) # calls PatchAutoEncoder's encode() method

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        #raise NotImplementedError()
        return super().decode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        #raise NotImplementedError()
        
        """
        NOTE: 
            - bc encode() will return output of dimension hwc (channel last), this dim will 
              be fed into bsq and bc not working with conv layers we dont worry about doing the hwc_to_chw(). Then 
              call PatchAutoEncoder's decoder which expects hwc dim format so this pipeline works (regarding dimensions)
            - also note, true dimension of input is batch x h x w x c
            - IMPORTANT to NOTE: 
                - Encode_index and decode_index are only used for autoregressive generation (so not BSQ). Do not use them in training. 
                - This is why in BSQ's forward() we call BSQ's encode and decode methods rather encode_index and decode_index
        """
        
        # Encode using PatchAutoEncoder's encode method 
        x=self.encode(x)
        
        # BSQ, perform linear down projection, normalization, binary quantize, then linear up projection
        x=self.bsq(x)
        
        # finally decode using PatchAutoEncoder's decode method 
        return self.decode(x), {}