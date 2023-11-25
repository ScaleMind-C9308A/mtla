def accuracy(logits, label):
    B, _ = logits.shape
    return (logits.argmax(dim=1) == label).sum().item() / (B)