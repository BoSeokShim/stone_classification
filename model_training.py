from matplotlib import pyplot as plt
from keras import initializers
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from WarmUpCosineDecayScheduler import WarmUpCosineDecayScheduler
import sys


##########################################################################
# training 세팅을 위한 코드
##########################################################################

# 교차검증 (cross-validation) 을 위해 나눠놓은 데이터셋 번호 (0~7까지)
# 총 8개의 DB가 있으며, 몇번 DB를 쓸것인지 선택함
dataset_no = 10

# 학습할 epoch (학습 시간)
epochs = 5

# 네트워크 이름 선택
network = 'inception_resnet' # ['inception_resnet, vgg, resnet50']

# 빠른 학습을 위한 hyper parameter multiplier
# (above 4 -> memory issue)
multiplier = 2
warmup_epoch = 5 # 초반 x epoch만큼 learning rate를 작게 설정
learning_rate_base = 0.002*multiplier # 기본 learning rate
batch_size = 16*multiplier # 배치 사이즈

################################################################################
# data loading (based on data generator)
################################################################################

train_data_dir = 'dataset/SET%d/training_set' % dataset_no
validation_data_dir = 'dataset/SET%d/test_set' % dataset_no
img_width, img_height = 224, 224

# image generator
train_datagen = ImageDataGenerator(
                    rotation_range=60,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    rescale = 1./255,
                    shear_range = 0.05,
                    zoom_range = 0.4,
                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
sample_count = training_set.samples
test_set = test_datagen.flow_from_directory(validation_data_dir,
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
# the number of samples
print(sample_count)

################################################################################
# build the network
################################################################################
if (network == 'inception_resnet'):
    input = Input(shape=(224, 224, 3))
    model = InceptionResNetV2(input_tensor=input, include_top=False, weights=None, pooling='max')

    x = model.output
    x = Dense(512, name='fully', kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(256, kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dense(1, activation='softmax', name='softmax')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(model.input, x)
    model.summary()
    print('Model loaded.')

elif (network == 'resnet50'):
    input = Input(shape=(224, 224, 3))
    model = ResNet50(input_tensor=input, include_top=False, weights=None, pooling='max')

    x = model.output
    x = Dense(512, name='fully', kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(256, kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dense(1, activation='softmax', name='softmax')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(model.input, x)
    model.summary()
    print('Model loaded.')

elif (network == 'vgg'):
    new_model = applications.VGG19(include_top=False, input_shape=(img_width, img_height, 3))

    model = Sequential()
    for layer in new_model.layers:
        model.add(layer)
    model.add(Flatten(input_shape=new_model.output_shape[1:]))
    model.add(Dense(128, activation='relu', kernel_initializer=initializers.he_uniform(seed=None)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer=initializers.he_uniform(seed=None)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print('Model loaded.')
else:
    print('wrong network type!')
    sys.exit()

################################################################################
# compile the model / training
################################################################################
model.compile(loss='binary_crossentropy',
              # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              # optimizer=optimizers.RMSprop(lr=2e-4),
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])


# Create the Learning rate scheduler.
# trick 1. Efficient training
# Compute the number of warmup batches.
steps_per_epoch_ = int(sample_count / batch_size)
total_steps = int(epochs * sample_count / batch_size)
warmup_steps = int(warmup_epoch * sample_count / batch_size)
val_steps = int(test_set.samples/batch_size)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=25)

# Train the model, iterating on the data in batches
history = model.fit_generator(training_set,
                              epochs=epochs,
                              steps_per_epoch = steps_per_epoch_,
                              verbose=1,
                              validation_data=test_set,
                              callbacks=[warm_up_lr]
                              #, class_weight = {0.0: 0.9, 1.0: 0.1}
                              )

model.save_weights('weight_SET%d_%s.h5' % (dataset_no, network))

################################################################################
# model evaluation
################################################################################
print("-- Evaluate --")
test_set.reset()
scores = model.evaluate_generator(test_set, steps=val_steps)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_ = range(1, len(acc) + 1)

val_str = 'Validation acc %0.4f' % scores[1]

# make result plot
fig = plt.figure()
ax1 = plt.subplot(121)
ax1.plot(epochs_, acc, 'bo', label='Training acc')
ax1.plot(epochs_, val_acc, 'b', label=val_str)
ax1.legend()
ax2 = plt.subplot(122)
ax2.plot(epochs_, loss, 'bo', label='Training loss')
ax2.plot(epochs_, val_loss, 'b', label='Validation loss')
ax2.legend()

# visualization
# plt.show()
fig.savefig('SET%d_result_%s.jpg' % (dataset_no, network))
plt.close(fig)

################################################################################
# model configuration
################################################################################

print(epochs_)
print(val_acc)
print(val_loss)

import json

file_path = "C:/Users/ShimBoSeok/Desktop/20-08-20_SET%d_config.json" % dataset_no

DEFAULTS = {
    "dataset_no": dataset_no,
    "network": network,
    "epochs": epochs,
    "multiplier": multiplier,
    "optimizer": {
        "warmup_epoch": warmup_epoch,
        "batch_size": batch_size,
        "type": "Adam",  # supported: SGD, Adam
        "learning_rate_base": learning_rate_base,
        "steps_per_epoch_": steps_per_epoch_,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "val_steps": val_steps,
    },
    "evaluation": {
        "val_acc": val_acc[epochs-1],
        "val_loss": val_loss[epochs-1],
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(file_path, defaults=DEFAULTS):
    with open(file_path, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config

