import keras.optimizers
import keras.regularizers as regularizers
from keras import losses
from keras.layers import Input, Dense, Add
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def torank4(X):
    return X.reshape((X.shape[0],28,28,1))

def torank2(i):
    (X, Y)=i
    return X.reshape((X.shape[0], 784)),Y

def onehot(X):
    y=np.zeros((X.shape[0],10),dtype=int,)
    y[range(X.shape[0]),X]=1
    return y

def load_data():
    #############
    # LOAD DATA #
    #############
    dataset = 'C:/Users/phulo/Downloads/mnist.pkl.gz'

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    # print(onehot(train_set[1]))
    print('... loading data')
    print(train_set[0].sum())
    print(onehot(train_set[1]).sum(axis=0))
    p1=np.random.permutation(range(train_set[0].shape[0]))
    p2=np.random.permutation(range(valid_set[0].shape[0]))
    p3=np.random.permutation(range(test_set[0].shape[0]))
    rval = {
        'X_train' : train_set[0][p1],
        'Y_train' : onehot(train_set[1][p1]),
        'X_valid' : valid_set[0][p2],
        'Y_valid' : onehot(valid_set[1][p2]),
        'X_test' : test_set[0][p3],
        'Y_test' : onehot(test_set[1][p3])
    }
    return rval

def MLP_MNIST(data, nb_epoch=20, batch_size=30, verb=0, **kwargs):
    hparams = {
        'activation': 'relu',
        'noeuds': [256,256,256],
        'epsilon': 6.8129e-05,
        'learning_rate': 0.013,
        'reg_l1': 0,
        'reg_l2': 0.0001,
        'n_couches': 3,
        'residual':False
    }
    hparams.update(kwargs)
    dim = data['X_train'].shape[1]
    if len(data['Y_train'].shape)==1:
        n_out = 1
    else :
        n_out = data['Y_train'].shape[1]

    input = Input(shape=(dim,))
    network = input
    init = keras.initializers.glorot_normal(seed=None)
    temp = input
    for i in range(hparams['n_couches']):
        if hparams['residual']&(i%3==0)&i!=0:
            concat = Add()([network,temp])
            temp = network
            network = Dense(hparams['noeuds'][i],
                            activation=hparams['activation'],
                            kernel_regularizer=regularizers.l1_l2(
                                hparams["reg_l1"],
                                hparams["reg_l2"]),
                            kernel_initializer=init,
                            )(concat)
        else :
            if i%3==1:
                temp = network
            network = Dense(hparams['noeuds'][i],
                            activation=hparams['activation'],
                            kernel_regularizer=regularizers.l1_l2(
                                hparams["reg_l1"],
                                hparams["reg_l2"]),
                            kernel_initializer=init,
                            )(network)

    network = Dense(n_out, activation='softmax',
                    kernel_regularizer=regularizers.l1_l2(
                        hparams["reg_l1"],
                        hparams["reg_l2"]),
                    kernel_initializer=init)(network)
    model = Model(inputs=input, outputs=network)

    obj = losses.categorical_crossentropy

    opt = keras.optimizers.Adam()
    hist = LossHistory()
    model.compile(optimizer=opt, loss=obj, metrics=['categorical_accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')


    model.fit_generator(map(torank2,datagen.flow(torank4(data['X_train']), data['Y_train'],batch_size=batch_size)),
                    steps_per_epoch=data['X_train'].shape[0]//batch_size,
                    nb_epoch=nb_epoch,
                    verbose=verb,
                    validation_data=(data['X_valid'], data['Y_valid']),
                    callbacks=[hist])
    #
    # model.fit(data['X_train'], data['Y_train'],
    #           epochs=nb_epoch,
    #           batch_size=batch_size,
    #           verbose=verb,
    #           shuffle=True,
    #           validation_data=(data['X_valid'], data['Y_valid']),
    #           callbacks=[hist]
    #           )
    loss = model.evaluate(data['X_test'], data['Y_test'], verbose=0)
    print(loss)
    return model, hist

def plot(models,s,leg):
    plt.figure(figsize=(16, 4), dpi=100)
    plt.subplot(121)
    for model,l in zip(models,leg):
        plt.plot(model.history.history[s],label=l)
    plt.title('training')
    plt.subplot(122)
    for model,l in zip(models,leg):
        plt.plot(model.history.history['val_'+s],label=l)
    plt.title('validation')
    plt.legend()
    plt.show()

def plot_hist(hists,leg):
    plt.figure(figsize=(16, 4), dpi=100)
    plt.subplot(121)
    for hist,l in zip(hists,leg):
        plt.plot(hist.losses,label=l)
    plt.title('training')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import tensorflow as tf
    with tf.device('/cpu:0'):
        data = load_data()
        dic ={
            'activation':'sigmoid',
            # 'residual':True,
            'noeuds':[256,256,256],
            'learning_rate':0.13
        }
        dic['n_couches']=len(dic['noeuds'])
        model, hist = MLP_MNIST(data,verb=2,**dic)

        # print(model.history.history['loss'])
        # print(model.history.history.keys())
        plt.subplot(221)
        plt.plot(model.history.history['val_loss'])
        plt.plot(model.history.history['loss'])
        plt.title('NLL')
        plt.subplot(222)
        plt.plot(model.history.history['val_categorical_accuracy'])
        plt.plot(model.history.history['categorical_accuracy'])
        plt.title('accuracy')
        plt.subplot(223)
        plt.plot(hist.losses)
        plt.plot(hist.val_losses)
        plt.title('loss per epoch')
        plt.show()