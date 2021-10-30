import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt

path_dir1 = './0/'
path_dir2 = './1/'
path_dir3 = './2/'
path_dir4 = './3/'
path_dir5 = './4/'

file_list1 = os.listdir(path_dir1)  # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
file_list3 = os.listdir(path_dir3)  # path에 존재하는 파일 목록 가져오기
file_list4 = os.listdir(path_dir4)
file_list5 = os.listdir(path_dir5)  # path에 존재하는 파일 목록 가져오기


file_list1_num = len(file_list1)
file_list2_num = len(file_list2)
file_list3_num = len(file_list3)
file_list4_num = len(file_list4)
file_list5_num = len(file_list5)
file_num = file_list1_num + file_list2_num + file_list3_num + file_list4_num + file_list5_num

# %% 이미지 전처리

num = 0;
all_img = np.float32(np.zeros((file_num, 224, 224, 3)))
all_label = np.float64(np.zeros((file_num, 1)))

for img_name in file_list1:
    img_path = path_dir1 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 0  # not detect
    num = num + 1

for img_name in file_list2:
    img_path = path_dir2 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 1  # 1stage
    num = num + 1

for img_name in file_list3:
    img_path = path_dir3 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 2  # 2stage
    num = num + 1


for img_name in file_list4:
    img_path = path_dir4 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 3  # 3stage
    num = num + 1

for img_name in file_list5:
    img_path = path_dir5 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 4  # 4stage
    num = num + 1

# 데이터셋 섞기(적절하게 훈련되게 하기 위함)
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)

all_label = all_label[indices]
all_img = all_img[indices]

# 훈련셋 테스트셋 분할
num_train = int(np.round(all_label.shape[0] * 0.8))
num_test = int(np.round(all_label.shape[0] * 0.2))

train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :]

train_label = all_label[0:num_train]
test_label = all_label[num_train:]

# %%
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

flatten_layer = Flatten()
dense_layer1 = Dense(128, activation='relu')
bn_layer1 = BatchNormalization()
dense_layer2 = Dense(5, activation=tf.nn.softmax)

model = Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    bn_layer1,
    dense_layer2,
])

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_img, train_label, epochs=10, batch_size=32, validation_data=(test_img, test_label))

# save model
model.save("hand_wash_resnet_model.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='testing acc')
plt.title('Training and testing accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='testing loss')
plt.title('Training and testing loss')
plt.legend()

plt.show()

print("Saved model to disk")