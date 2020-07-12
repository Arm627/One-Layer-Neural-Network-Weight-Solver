from autoWeightSolver_v1 import AutoWeightSolver
import tensorflow as tf
import matplotlib.pyplot as plt
import time, heapq
import numpy as np

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

BATCH_SIZE = 64
EPOCHS = 5 

num_samples = len(train_images)
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

times = []
labels = []
start = time.time()
model = AutoWeightSolver(X, Y, add_bias=True)
model.weightSolver(activation='sigmoid')
print(time.time() - start)
times.append(time.time() - start)

model = build_ann_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(1e-1), metrics=['accuracy'])
start = time.time()
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
times.append(time.time() - start)
print('Elapsed time: {}'.format(time.time() - start))

labels = ['Reversed NN', 'Standard NN (5 epochs)']
plt.bar(labels, times)
plt.xlabel('NN Type')
plt.ylabel('Training Speed (seconds)')
plt.show()
plt.savefig('times_result.png')