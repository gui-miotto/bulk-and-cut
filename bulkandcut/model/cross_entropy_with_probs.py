import torch
import snorkel.classification

class CrossEntropyWithProbs(torch.nn.Module):
    """Cross entropy for soft labels. PyTorch, unlike TensorFlow or Keras, requires this
    workaround because CrossEntropyLoss demands that labels are given in a LongTensor.
    """

    def __init__(self, weight: "Optional[torch.Tensor]" = None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):  #TODO: copy code to reduce dependencies
        loss = snorkel.classification.cross_entropy_with_probs(
            input=input,
            target=target,
            weight=self.weight,
            reduction=self.reduction,
            )
        return loss
