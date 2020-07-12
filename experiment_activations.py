from autoWeightSolver_v1 import AutoWeightSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, heapq

activations = ['sigmoid', 'softmax', 'lrelu']

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
start = time.time()
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (train_images/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (test_images/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)

num_samples = len(train_images)

accuracies = []
for act in activations:
    indexes = np.random.randint(0, len(train_images), num_samples)

    new_train_labels = np.empty((len(train_labels), len(classes)))
    for i in range(len(train_labels)):
        new_train_label = np.full((len(classes)), 0.001)
        new_train_label[train_labels[i]] = 0.99
        new_train_labels[i] = new_train_label

    new_test_labels = np.empty((len(test_labels), len(classes)))
    for i in range(len(test_labels)):
        new_test_label = np.full((len(classes)), 0.001)
        new_test_label[test_labels[i]] = 0.99
        new_test_labels[i] = new_test_label

    input_size = 784
    X = np.empty((num_samples, 1, input_size))
    Y = np.empty((num_samples, len(classes), 1))
    for i in range(len(indexes)):
        label = np.expand_dims(new_train_labels[indexes[i]], -1)
        img = np.expand_dims(train_images[indexes[i]].flatten(), 0)
        X[i] = img 
        Y[i] = label

    model = AutoWeightSolver(X, Y, add_bias=True)
    model.weightSolver(activation=act)
    print('Elapsed time for {} training examples:'.format(num_samples), time.time() - start, 'Using bias =', repr(model.add_bias))

    num_correct = 0
    total = len(test_images)
    for i in range(len(test_images)):
        label = np.argmax(new_test_labels[i])
        img = test_images[i].flatten()
        predictions = model.evaluate(img).flatten()
        n_largest = heapq.nlargest(3, range(len(predictions)), key=predictions.__getitem__)
        if n_largest[0] == label:
            num_correct += 1
    accuracy = num_correct / total
    accuracies.append(accuracy)

plt.bar(activations, accuracies)
plt.xlabel('Activation')
plt.ylabel('Accuracy')
plt.savefig('activation_result.png')
plt.show()