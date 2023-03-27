def random_walk(n_samples: int = 10000, begin_value: float=None):
    import numpy as np
    if begin_value is None:
        begin_value = np.random.rand()
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(out[-1] + (np.random.rand()*2 - 1))
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.plot(random_walk())
    plt.show()
