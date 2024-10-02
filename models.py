# models.py

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import ConvLSTM2D, TimeDistributed, Dense, Flatten
from tensorflow.keras.models import Sequential
from config import XCEPTION_MODEL_PATH, FACE_XRAY_MODEL_PATH, CONV_LSTM_MODEL_PATH, WAVEFAKE_MODEL_PATH, RESNET_AUDIO_MODEL_PATH

def load_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
    output = Dense(1, activation='sigmoid')(base_model.output)
    xception_model = Model(inputs=base_model.input, outputs=output)
    if XCEPTION_MODEL_PATH and os.path.exists(XCEPTION_MODEL_PATH):
        xception_model.load_weights(XCEPTION_MODEL_PATH)
    return xception_model

def load_face_xray_model():
    if FACE_XRAY_MODEL_PATH and os.path.exists(FACE_XRAY_MODEL_PATH):
        face_xray_model = load_model(FACE_XRAY_MODEL_PATH)
    else:
        face_xray_model = load_xception_model()
    return face_xray_model

def load_conv_lstm_model():
    base_model = load_xception_model()
    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(None, 299, 299, 3)))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    if CONV_LSTM_MODEL_PATH and os.path.exists(CONV_LSTM_MODEL_PATH):
        model.load_weights(CONV_LSTM_MODEL_PATH)
    return model

def load_wavefake_model():
    if WAVEFAKE_MODEL_PATH and os.path.exists(WAVEFAKE_MODEL_PATH):
        wavefake_model = load_model(WAVEFAKE_MODEL_PATH)
    else:
        wavefake_model = Sequential()
        wavefake_model.add(Dense(1, activation='sigmoid', input_shape=(16000, 1)))
    return wavefake_model

def load_resnet_audio_model():
    if RESNET_AUDIO_MODEL_PATH and os.path.exists(RESNET_AUDIO_MODEL_PATH):
        resnet_audio_model = load_model(RESNET_AUDIO_MODEL_PATH)
    else:
        resnet_audio_model = Sequential()
        resnet_audio_model.add(Dense(1, activation='sigmoid', input_shape=(128, None, 1)))
    return resnet_audio_model

