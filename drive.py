import socketio
import eventlet
from keras.models import load_model
import base64
from flask import Flask
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()

app = Flask(__name__)  # '__main__'

speed_limit = 10


@sio.on('connect')
def connect(session_id, environment):
    print('Connected!')
    send_control(0, 0)


@sio.on('telemetry')
def telemetry(session_id, data_from_simulator):
    speed = float(data_from_simulator['speed'])
    image = Image.open(BytesIO(base64.b64decode(data_from_simulator['image'])))
    image = np.asarray(image)
    image = image_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


def image_preprocess(img):
    img = img[60:135, :, :]  # Cropping the height of the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Using YUV instead of GRAYSCALE because we will use the NVidia Archicteture for Neural Networks
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))  # Recommended size for the NVidia Architecture
    img = img / 255  # Normalization
    return img


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
