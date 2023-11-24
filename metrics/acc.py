def accuracy(logits, label):
    return (logits.argmax(dim=1) == label).sum().item()