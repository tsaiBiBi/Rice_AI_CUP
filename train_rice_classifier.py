from data import load_data, load_data
import numpy as np
import cv2
# from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    # plt.ylabel('Accuracy')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i]],
          'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_test[i],(IMAGE_SIZE,IMAGE_SIZE,3)))
    plt.show()
    for j in range(8):
        print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

IMAGE_SIZE = 32
WINDOW_SIZE = 32
# input
start = time.time()
(x_train,y_train),(x_test,y_test)=load_data()

x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)


print(time.time() - start)
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

y_train_OneHot = keras.utils.to_categorical(y_train)
y_test_OneHot = keras.utils.to_categorical(y_test)

# model v1
img_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='img_input')
hidden = layers.Conv2D(filters=12, kernel_size=(3,3), strides=1, activation='relu', name='hidden', padding='same')(img_input)
pool = layers.MaxPooling2D(pool_size=(2, 2), name='pool')(hidden)
hidden_ft = layers.Flatten()(pool)

hidden2 = layers.Dense(512, activation='sigmoid', name='hidden2')(hidden_ft)
dropout2 = layers.Dropout(rate=0.25)(hidden2)

hidden3 = layers.Dense(512, activation='relu', name='hidden3')(dropout2)
dropout3 = layers.Dropout(rate=0.25)(hidden3)

hidden4 = layers.Dense(512, activation='relu', name='hidden4')(dropout3)
dropout4 = layers.Dropout(rate=0.25)(hidden4)

outputs = layers.Dense(2, activation='softmax', name='Output')(dropout4)
model = keras.Model(inputs=img_input, outputs=outputs)

# model v2
# model = VGG16(weights=None, include_top=False, input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3))
# x = model.output
# x = Flatten()(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(2, activation='sigmoid')(x)

# model = Model(inputs=model.input, outputs=x)
# model.compile(
#     Adam(lr=1e-4),
#     loss='binary_crossentropy',
#     metrics=['acc'],
# )
# train_history=model.fit(x_train_normalize, y_train_OneHot,
#                         validation_split=0.2,
#                         epochs=10, batch_size=128, verbose=1)

print(model.summary())
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='./img/model_v1.png', show_shapes=True)

start = time.time()
# train
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x_train_normalize, y_train_OneHot,
                        validation_split=0.2,
                        epochs=20, batch_size=128, verbose=1)
print("Execution Time: ", time.time() - start)

# show training history
show_train_history('accuracy','val_accuracy')
show_train_history('loss','val_loss')
# show_train_history('acc','val_acc')
# show_train_history('loss','val_loss')

# Step 6. 評估模型準確率
scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=0)
print(scores[:10])

# 進行預測

prediction=model.predict(x_test_normalize)
prediction=np.argmax(prediction,axis=1)
prediction[:10]

print("prediction")
print(prediction.shape)
print(prediction)

# 查看預測結果

label_dict={
  0:'Not', 
  1:'Rice',
}
plot_images_labels_prediction(x_test,y_test,prediction,0,10)

# # 查看預測機率
# Predicted_Probability=model.predict(x_test_normalize)
# show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,0)

# Step 8. Save Weight to h5 

# model.save_weights("./riceClassifier_model.h5")
model.save_weights("./model.h5")
print("Saved model to disk")