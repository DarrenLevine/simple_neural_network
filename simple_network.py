"""A minimalistic neural network implementation that can learn the MNIST data set and plot its training progress"""
import numpy as np
import plotting as plt


class Layer():
    def __init__(self, in_size, out_size):
        # Using "He" weight initialization
        self.weights = np.float32(np.random.normal(0., np.sqrt(2 / float(in_size)), size=(out_size, in_size)))
        self.biases = np.float32(np.zeros((out_size, 1))) + 0.1


class SimpleNetwork():

    def __init__(self, **kargs) -> None:
        self.layers = kargs.get('layers', [64, 64, 64, 10])
        self.learning_rate = kargs.get('learning_rate', 0.1)
        self.transfer_func = kargs.get('transfer_func', lambda x: x * (x > 0.))  # defaults to ReLU
        self.d_transfer_func = kargs.get('d_transfer_func', lambda x: x > 0.)  # defaults to dRelU
        self.network = [Layer(i, o) for i, o in zip([self.layers[0]] + self.layers[:-1], self.layers)]
        self.plotter = plt.Plotter()

    def train(self, training_data):
        all_indexes = list(range(len(training_data)))
        batch_size = len(all_indexes) // 10
        epochs = 0
        training = True
        figure_title = 'Training progress (close this plot to stop training)'
        while training:
            epochs += 1
            training_steps = 0
            passes = 0.
            np.random.shuffle(all_indexes)
            for row_index in all_indexes[:batch_size]:
                training_steps += 1
                if not self.plotter.is_open(figure_title):
                    training = False
                    break

                # get input with target output values
                target, act = training_data[row_index]

                # forward pass
                activations = []
                for n in self.network:
                    activations += [act]
                    act = self.transfer_func(np.dot(n.weights, act) + n.biases)

                # calculate error
                error = act - target
                passes += (np.where(act == act.max())[0][0] == np.where(target == target.max())[0][0])

                # reverse pass (backprop / weight updating)
                for i in reversed(range(len(self.network))):
                    propagated_error = np.dot(self.network[i].weights.T, error) * self.d_transfer_func(activations[i])
                    self.network[i].weights -= self.learning_rate * np.dot(error, activations[i].T) / len(target)
                    self.network[i].biases -= self.learning_rate * np.sum(error, axis=1, keepdims=True) / len(target)
                    error = propagated_error

            pass_ratio = passes / training_steps
            print(f'training epoch {epochs} pass_ratio: {pass_ratio:.2f}')
            self.plotter.plot(figure_title, epochs, pass_ratio, x_label='Epoch', y_label='Pass Ratio')
            self.inspect_weights(pause_time=0.1)

    def save(self, file='pre_trained_network_weights.npy'):
        np.save(file, self.network, allow_pickle=True)

    def load(self, file='pre_trained_network_weights.npy'):
        self.network = np.load(file, allow_pickle=True)

    def inspect_weights(self, pause_time=None):
        self.plotter.image('Layer Weight Values',
                           [L.weights for L in self.network],
                           title=[f'layer {i}' for i in range(len(self.network))],
                           pause_time=pause_time)

    def inference(self, activation):
        for n in self.network:
            activation = self.transfer_func(np.dot(n.weights, activation) + n.biases)
        return activation

    def test(self, test_data):
        passes = 0.
        for target, act in test_data:
            act = self.inference(act)
            passes += (np.where(act == act.max())[0][0] == np.where(target == target.max())[0][0])
        print(f'testing pass_ratio: {passes / len(test_data):.2f}')

    def mnist_demo(self):
        MNIST_data = np.load('mnist_mini_dataset.npy')  # low resolution 8x8 pixel MNIST data
        MNIST_data = [  # separate data into an array of tuples, with two items each:
            (np.array([x[:10]]).T,  # label: an array of 0/1 values, the index of the 1 is the numbers value
             np.array([x[10:]]).T)  # data: the image grey-scale values for the 2d image of a number
            for x in MNIST_data]

        # % to view one of the images of a number, uncomment the following:
        # label_data, image_data = MNIST_data[0]
        # self.plotter.image(f'Label vector: {str(label_data.T)}', image_data.reshape((8, 8)))

        # % separate the full data set into a training data set, and a testing data set
        training_data_to_test_data_ratio = 0.8
        cut_index = int(len(MNIST_data) * training_data_to_test_data_ratio)
        training_data, test_data = MNIST_data[:cut_index], MNIST_data[cut_index:]

        # % comment out/in the following lines to play with demo network:
        # self.train(training_data)
        # self.save()
        self.load()
        self.test(test_data)
        self.inspect_weights()


if __name__ == "__main__":
    np.random.seed(987766)
    SimpleNetwork().mnist_demo()
