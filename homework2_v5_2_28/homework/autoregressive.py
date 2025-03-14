import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


# Dont have to write code in here, these are just abstract methods
class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        #raise NotImplementedError()
        
        
        """
        Transformer layer
        - https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        - torch has premade encoder and decoder only layers (this corresponds to as seen in the transformer paper of the encoder and decoder blocks)
        - we're recommended to use decoder only but use torch's TransformerEncoderLayer which follows the encoder architecture in paper 
        so the MultiHeadAttention self attention then feedforward (linear, relu, linear) with the layernorms and residual connections
        - Reasons why doing using torchencoderlayer:
            - TransformerEncoderLayer includes self-attention, and when used with a causal mask, it behaves like a decoder.
            - both TransformerEncoderLayer and TransformerDecoderLayer works. 
                - But TransformerEncoderLayer:
                    - is more memory efficient (doesn't require encoder output), since we want decoder-only model
                    - it's simpler, doesn't support cross attention - which we don't need
        """
        
        # params corresponding to transformer layer
        num_layers=2
        num_heads=8
        dim_feedforward=1024 # default was 2048 which is too much , this corresponds to the linear(d_model, dim_feedforward), relu(), linear(dim_feedforward, d_model)
        dropout=0.1 # 0.1 used in transformer paper
        # norm_first=True: in deeplearning lecture, doing norm first before attention was better but not done in orig transformer paper
        activation="relu" # can change to use gelu
        # we set batch_first=True bc our input's first dimension is batch
        

        self.network = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_latent),
            *[TransformerEncoderLayer(d_model=n_tokens, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True, batch_first=True) for _ in range(num_layers)],
            torch.nn.Linear(n_tokens, d_latent),
            torch.nn.Softmax(dim=-1), # softmax bc want to output probabities
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #raise NotImplementedError()
        
        # NOTE: input is size (Batch x height x width) and are the compressed tokenized version of images generated from our BSQ model
        # -> output of BSQ model is the output of last layer being the decoder from autoencoder model being UnpatchifyLinear(): (B, H * patch_size, W * patch_size, 3)
        # -> note UnpatchifyLinear() returns the original image so (Batch x Original Height x Original Width x 3 Color Channels)
        """
        Mask
        - mask we're using (already made mask): 
        - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.generate_square_subsequent_mask
        - creates a square mask matrix of size (sz x sz) where sz is the sequence length or number of tokens. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0)
        - -inf marks invalid locations so this mask prevents from looking at future locations (ith row is current location and jth column are the possible locations)
        example output of sz=5:
            torch.Size([5, 5]) 
            tensor([[0., -inf, -inf, -inf, -inf],
                    [0., 0., -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., 0.]])
        """
        sz=x.shape[1] * x.shape[2] # height * width is the number of tokens or sequenceLength bc this is the total num of pixels in an image
        mask=torch.nn.Transformer.generate_square_subsequent_mask(sz=sz) 
        
        """
        - The shape of the input before flattening is shape of x torch.Size([2560, 100, 150, 3])
        - The shape of the input after flattening is shape of x torch.Size([2560, 45000])
        """
        x=torch.flatten(x)
    
    # This is for part 4 (the generate part of the assignment)
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        raise NotImplementedError()




"""
Transformer Notes:
- Attention is basic building block of transformer architecture, it is similar to convolution but instead works with a variable length size input
- Attention is a set operator that learns to reason about the structure of a set of elements
- Purpose of attention: To reason over sets of elements and relate individual elements with one another 
- Attention operator uses a query, set of keys, and set of values
    - Each value has a key
    - Each word is a value which is the meaning of the word
    - Query allows us to exchange information between the keys/values with the query
    - Query will measure distance from itself to all keys (dot product q and k) then scaled by C being # keys, exponentiate it, then normalize it
    - This ends up being the softmax of single query and distance to all keys
    - Alpha i is the resulting weight that will all sum to 1 and are always positive
    - Then compute weighted average (omega o)
    - If key is close by then value will have high weight and otherwise will have weight close to 0 or negative having little effect on weighted sum
    - Attention is most likely written as matrix operator where we dont have 1 query but a set of n queries, will compute distances between all queries 
    and all keys, then normalize, then compute softmax (property of softmax is all will sum to 1) and then will retrieve weighted avg of values v
        - IMPORTANT
        - Notice what is being done:
        - Take dot product of Q query and K keys then divide by sqrt C to scale, then softmax is done to normalize (values between 0 and 1 that sum to 1)
        - Then take dot product with V values ( practically weighting importance of values in V based on results from Q*K)
        
- preprocessing for language models:
    - split sentence into parts: characters, words, tokens
    - then embed each part (translate the part (set of chars) to a represnetable vector of numbers
    - Embedding associates a certain set of feature vectors with a specific input (basically translates the part/token into something the network understands being numbers then attention is fed this to relate the #s to one another)
        - Take sentence and parse it
        - Turn parts into #s through embedding
        - Feed embedding into network that will learn to relate #s to one another
        
- Cross Attention and Self Attention    
    - Cross Attention: Query comes from its own input and keys and values come from the same input -> can think about trying to relate spanish words to english words
    - Self Att: Query, keys, and values come from the same input
    - In cross attention, half input goes to query and other half to keys and values
    - Self attention is used in NLP and cross attention is used in computer vision tasks 

- Multi Head Attention
    - Almost no one uses basic attention operator, multi head attention is what people actually use
    - Problem for wanting to deal with arbitrary sequences: An attention always attends more to itself than anything else
    - Basically what the issues of self attention is trying to say is that the attention head when observing an ith row and comparing to other columns, will pay attention to itself more than other tokens bc the dot product has larger value
    - Now apply a linear layer to each input (query, key, value) (creating weight matrices) which allows us to have arbitrary values for query keys and values and then compute attention
    - This trick allows attentions to attend more to other elements rather itself which allows the attention to look at say arbitrary spots in the image
    - Adding the weights made the attention as expressive as a 1x1 convolution but issue is attention operator uses the same attention matrix so cannot gather information from multiple disjointed places, or if it can, it has to average the information together. 
    - SO solution is multi head attention
        - is concatentaiton of multiple attention layers (heads)
    - work flow of MHA
        - h heads (attention layers)
        - each starts with input key, value, query and performs linear projection (linear layer) then is fed into attention operator (scaled Query * key) * value
        - then concat the heads
        - finally perform single linear projection (layer) on concatenated outputs
    - self attention with weights generalizes a 1x1 conv2D
    - MHA with h heads is more expressive than a convolution with sqrt(h) x sqrt(h) kernel size
    
    - typically use 8 or more heads

- The attention scores, computed based on the query and key dot product, are used to weight the value vectors. 
- Higher attention scores mean that the corresponding values are more important for the output.


- Attention with weights is permutation invariant meaning
- Can take bunch of inputs to attention keep queries same , permute all keys and values in arbitrary order, as long as you permute them the same, the output of the attention is the same
- This is a problem bc now we cant reason to the left and right of a location in a sentence. If we use an image, we cannot reason about any spatial arrangement
- Spatial location matters bc ordered operations relies on spatial location and say reshuffling of parts of pics changes meaning
- Solution: add Positional embedding to input (there are different kinds of positional embeddings)
    - absolute, sinusoidal, learnable, relative, rotary
    
- Transformer architecture is compact and computationally effective way to model set operations or even operations over arbitrarily sized inputs
- At its core uses multi head attention and then uses positional embedding to get rid of permutation invariance 
 and combine with multi layer perceptron MLP Multi Layer Perceptron(linear followed by non linear sets of layers)
- add layer norm and residual connections to avoid vanishing gradients
- Look at lecture 6.6 deep learning for visual of architecture
- Placing normalization before transformer (MHA) is computationally faster and better (not done in original transformer paper)
- Transformer layer=MHA(multi head attention) + MLP (multi layer perceptron) + LN (layer norm) + residual connections

lecture 6.8 deep learning
- Transformer used for auto regressive prediction: predicting one token at a time and each next word is conditioned on the previous predictions (what has seen so far)
- By using masking, we force the transformer to only look at words that came before the current token being predicted
- Vanilla transformer has attention to all inputs so it could easily cheat by looking at the token in the input at the current location and keep predicting that
- To mask you add a mask matrix to the attention operator softmax( (Q*K)/sqrt(C) + MaskMatrix) * V
- Mask matrix sets certain elements in the softmax to 0 then attention only goes backwards (only look at tokens before the current token being predicted)
- TYPES OF TRANSFORMERS:
    - Decoder only: masked auto regressive prediction
    - Encoder only: no prediction, just understanding

**** Example 6.9 Deep Learning Good Example. Look at Understnading doc and code
**** Also look at my annotated Transformer Paper

Masking Done in Transformer Paper:
- Masking here prevents the ith token/position being predicted from looking at future (subsequent) positions. 
- Also the output embeddings being shifted by 1 prevents the ith position from cheating by looking at it's ith position 
- (only look at positions less than i)
- negative infinity corresponds to illegal locations/connections 
-> can view a matrix mask as connections. Rows are the ith location and columns are jth location. So if an entry is -inf, then at location i, you cannot attend to location j

- Transformer paper uses learned embedding


"""