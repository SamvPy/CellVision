
def log_shape(name, tensor, trace=True):
    if trace:
        print(f"{name}: {tuple(tensor.shape)}")
    return tensor
