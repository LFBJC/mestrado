import numpy as np


def random_walk(n_samples: int = 10000, begin_value: float=None):
    import numpy as np
    if begin_value is None:
        begin_value = np.random.rand()
    out = [begin_value]
    for step in range(n_samples-1):
        out.append(out[-1] + (np.random.rand()*2 - 1))
    return out


def aggregate_data_by_chunks(data, chunk_size: int = 10):
    import numpy as np
    box_plots = []
    for chunk in range(int(np.ceil(len(data)/chunk_size))):
        chunk_data = data[chunk*chunk_size:chunk*chunk_size + chunk_size]
        box_plots.append({
            'whislo': min(chunk_data),
            'q1': np.quantile(chunk_data, 0.25),
            'med': np.quantile(chunk_data, 0.5),
            'q3': np.quantile(chunk_data, 0.75),
            'whishi': max(chunk_data)
        })
    return box_plots


def plot_single_box_plot_series(box_plot_series):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bxp(box_plot_series, showfliers=False)
    plt.show()


def plot_multiple_box_plot_series(box_plot_series):
    import matplotlib.pyplot as plt
    if len(box_plot_series) > 1:
        assert all([len(x) == len(box_plot_series[0]) for x in box_plot_series[1:]])
        # positions = list(range(len(box_plot_series[0])))*len(box_plot_series)
        colors = ['green', 'red', 'blue']
        if len(box_plot_series) > 3:
            import random
            for _ in range(len(box_plot_series) - 3):
                r = lambda: random.randint(0, 255)
                colors.append('#%02X%02X%02X' % (r(), r(), r()))
        fig, ax = plt.subplots()
        for i, b_series in enumerate(box_plot_series):
            ax.bxp(b_series, boxprops={'color': colors[i]}, showfliers=False)
        plt.show()


if __name__ == "__main__":
    plot_single_box_plot_series(aggregate_data_by_chunks(random_walk(), 100))
    plot_multiple_box_plot_series([aggregate_data_by_chunks(random_walk(), 100), aggregate_data_by_chunks(random_walk(), 100)])
