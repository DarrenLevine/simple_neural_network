from matplotlib import pyplot as plt


class PlotData():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.plot_handle = None
        self.fig = None
        self.ax = None
        self.im = None


class Plotter():
    def __init__(self) -> None:
        self.data = {}

    def get_data(self, name):
        if name not in self.data:
            self.data[name] = PlotData()
        return self.data[name]

    def is_open(self, title):
        data = self.get_data(title)
        return data.fig is None or plt.fignum_exists(data.fig.number)

    def plot(self, title, x, y, x_label=None, y_label=None):
        data = self.get_data(title)
        data.x += [x]
        data.y += [y]
        if len(data.x) > 1:
            if data.fig is None:
                data.fig, data.ax = plt.subplots(1, 1, figsize=(10, 5))
                data.plot_handle = plt.plot([], [], 'r-')[0]
                data.ax.set_title(title)
                data.ax.set_ylim(0., 1.)
                data.ax.set_xlabel(x_label)
                data.ax.set_ylabel(y_label)
                plt.show(block=False)
            data.plot_handle.set_data(data.x, data.y)
            data.ax.set_xlim(data.x[0], data.x[-1])
            plt.pause(0.1)

    def image(self, figure_title, img, title=None, pause_time=None):
        data = self.get_data(figure_title)
        if not isinstance(img, list):
            img = [img]
        creating_plot = data.fig is None or len(img) != len(self.im)
        if creating_plot:
            if data.fig is not None:
                plt.close(data.fig)
            data.fig, data.ax = plt.subplots(1, len(img), figsize=(10, 5))
            self.im = [None for _ in range(len(img))]
        if len(img) == 1:
            data.ax = [data.ax]
        data.fig.suptitle(figure_title)
        if title is None:
            title = [None] * len(img)
        if creating_plot:
            x_limits = []
            y_limits = []
            for i, c in enumerate(img):
                data.ax[i].set_title(title[i])
                self.im[i] = data.ax[i].imshow(c, cmap='gray_r')
                x_limits += [data.ax[i].get_xlim()]
                y_limits += [data.ax[i].get_ylim()]
            max_x_lim = max(x_limits)
            max_y_lim = max(y_limits)
            for i, _ in enumerate(img):
                data.ax[i].set_axis_off()
                data.ax[i].set_xlim(max_x_lim)
                data.ax[i].set_ylim(max_y_lim)
        else:
            for i, c in enumerate(img):
                self.im[i].set_data(c)
                data.fig.canvas.flush_events()
        if pause_time is None:
            plt.show()
        else:
            plt.pause(pause_time)
