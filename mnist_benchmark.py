import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

BATCH_SIZE = 16
EPOCHS = 5

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (np.expand_dims(train_images, axis=-1)/255).astype(np.float32).reshape((60000, 784))
print(train_images.shape)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32).reshape((10000, 784))
test_labels = (test_labels).astype(np.int64)

print(train_images.shape, train_labels.shape)

def build_ann_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax, name='dense2'))
    return model 

model = build_ann_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(1e-1), metrics=['accuracy'])
start = time.time()
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
print('Elapsed time: {}'.format(time.time() - start))
'''
dataset_size = [10, 100, 500, 1000, 5000, 10000, 30000, len(train_images)]
total_accuracy = []
for size in dataset_size:
    model = build_ann_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(1e-1), metrics=['accuracy'])
    BATCH_SIZE = 64
    EPOCHS = 5
    indexes = np.random.randint(0, len(train_images), size)
    new_train_images = []
    new_train_labels = []
    for i in indexes:
        new_train_images.append(train_images[i])
        new_train_labels.append(train_labels[i])
    model.fit(np.array(new_train_images), np.array(new_train_labels), batch_size=BATCH_SIZE, epochs=EPOCHS)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    total_accuracy.append(test_acc)
    print('Test accuracy:', test_acc)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(dataset_size, total_accuracy, label='1 Layer Regular NN')
ax1.plot(dataset_size, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], label='1 Layer Inverse NN (Our model)')
plt.legend(loc='bottom right')
plt.show()
'''