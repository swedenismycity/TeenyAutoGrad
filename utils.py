def fetch_mnist():
    path = r'C:\Users\linnb\Projects\autograd\MNIST_ORG\\'
    datasets = ["t10k-images.idx3-ubyte",
                "t10k-labels.idx1-ubyte",
                "train-images.idx3-ubyte",
                "train-labels.idx1-ubyte"]

    import numpy as np
    X_train = np.array(list(open(path+datasets[2], 'rb').read()))[16:].reshape((-1,28,28))
    Y_train = np.array(list(open(path+datasets[3], 'rb').read()))[8:]
    X_test = np.array(list(open(path+datasets[0], 'rb').read()))[16:].reshape((-1,28,28))
    Y_test = np.array(list(open(path+datasets[1], 'rb').read()))[8:]
    return X_train, Y_train, X_test, Y_test
