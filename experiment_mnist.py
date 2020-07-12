from autoWeightSolver_v1 import AutoWeightSolver
import tensorflow as tf
import matplotlib.pyplot as plt
import time, heapq
import numpy as np

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
start = time.time()
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (train_images/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (test_images/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)

def build_ann_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))
    return model 

dataset_size = [10, 100, 500, 1000, 5000, 10000, 30000, len(train_images)]
reverse_accuracy = []
original_accuracy = []

BATCH_SIZE = 64
EPOCHS = 5 

for s in dataset_size:
    num_samples = s
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
    model.weightSolver(activation='sigmoid')
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
    reverse_accuracy.append(accuracy)
    print(accuracy)

    new_train_images = []
    new_train_labels = []
    for i in indexes:
        new_train_images.append(train_images[i])
        new_train_labels.append(train_labels[i])
    model = build_ann_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(1e-1), metrics=['accuracy'])
    start = time.time()
    model.fit(np.array(new_train_images), np.array(new_train_labels), batch_size=BATCH_SIZE, epochs=EPOCHS)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)
    original_accuracy.append(test_acc)
    print('Elapsed time: {}'.format(time.time() - start))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(dataset_size, original_accuracy, label='1 Layer Regular NN (5 Epochs)')
ax1.plot(dataset_size, reverse_accuracy, label='1 Layer Reverse  Technique NN (My model)')
plt.xlabel('Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='bottom right')
plt.savefig('result.png')
plt.show()