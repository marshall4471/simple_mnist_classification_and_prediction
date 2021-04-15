#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import tensorflow_datasets as tfds


# In[4]:


mnist_bldr = tfds.builder('mnist')


# In[5]:


mnist_bldr.download_and_prepare()


# In[6]:


datasets = mnist_bldr.as_dataset(shuffle_files=False)


# In[7]:


mnist_train_orig = datasets['train']


# In[8]:


mnist_test_orig = datasets['test']


# In[9]:


BUFFER_SIZE = 10000


# In[10]:


BATCH_SIZE = 64


# In[11]:


NUM_EPOCHS = 20


# In[13]:


mnist_train = mnist_train_orig.map(
   lambda item: (tf.cast(item['image'], tf.float32)/255.0,
                 tf.cast(item['label'], tf.int32)))


# In[14]:


mnist_test = mnist_test_orig.map(
     lambda item: (tf.cast(item['image'], tf.float32)/255.0,
                   tf.cast(item['label'], tf.int32)))


# In[15]:


tf.random.set_seed(1)


# In[16]:


mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
                                  reshuffle_each_iteration=False)


# In[18]:


mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)


# In[19]:


mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)


# In[20]:


model = tf.keras.Sequential()


# In[22]:


model.add(tf.keras.layers.Conv2D(
          filters=32, kernel_size=(5, 5),
          strides=(1, 1), padding='same',
          data_format='channels_last',
          name='conv_1', activation='relu'))


# In[24]:


model.add(tf.keras.layers.MaxPool2D(
     pool_size=(2, 2), name='pool_1'))


# In[29]:


model.add(tf.keras.layers.Conv2D(
      filters=64, kernel_size=(5, 5),
      strides=(1, 1), padding='same',
      name='conv_2', activation='relu'))


# In[30]:


model.add(tf.keras.layers.MaxPool2D(
     pool_size=(2, 2), name='pool_2'))


# In[31]:


model.compute_output_shape(input_shape=(16, 28, 28, 1))


# In[32]:


model.add(tf.keras.layers.Flatten())


# In[33]:


model.compute_output_shape(input_shape=(16, 28, 28, 1))


# In[34]:


model.add(tf.keras.layers.Dense(
      units=1024, name='fc_1',
      activation='relu'))


# In[35]:


model.add(tf.keras.layers.Dropout(
          rate=0.5))


# In[36]:


model.add(tf.keras.layers.Dense(
          units=10, name='fc_2',
          activation='softmax'))


# In[37]:


tf.random.set_seed(1)


# In[38]:


model.build(input_shape=(None, 28, 28, 1))


# In[39]:


model.compile(
     optimizer=tf.keras.optimizers.Adam(),
     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
     metrics=['accuracy'])


# In[40]:


history = model.fit(mnist_train, epochs=NUM_EPOCHS,
                    validation_data=mnist_valid,
                    shuffle=True)


# In[42]:


import matplotlib.pyplot as plt


# In[43]:


import numpy as np


# In[44]:


hist = history.history


# In[45]:


x_arr = np.arange(len(hist['loss'])) + 1


# In[46]:


fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<',
        label='Validation acc.')
ax.legend(fontsize=15)
plt.show()


# In[47]:


test_results = model.evaluate(mnist_test.batch(20))


# In[48]:


print('Test ACC.: {:.2f}\%'.format(test_results[1]*100))


# In[49]:


batch_test = next(iter(mnist_test.batch(12)))


# In[50]:


preds = model(batch_test[0])


# In[51]:


tf.print(preds.shape)


# In[52]:


preds = tf.argmax(preds, axis=1)


# In[53]:


print(preds)


# In[54]:


fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    img = batch_test[0][i, :, :, 0]
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, '{}'.format(preds[i]),
    size=15, color='blue',
    horizontalalignment='center',
    transform=ax.transAxes)
plt.show()

