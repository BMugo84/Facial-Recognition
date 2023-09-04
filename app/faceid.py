# import kivy dependencies 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# import other dependancies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# build app and layout 
class CamApp(App):

    def build(self):
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", size_hint=(1, .1))
        self.verification = Label(text="verification Uninitiated", size_hint=(1,.1))

        # add items to layout 
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # load tf model
        self.model = tf.keras.models.load_model('siamesemodel.pkl', custom_objects={'L1Dist':L1Dist})

        # setup capture
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    # run continuously to get webcam feed
    def update(self, *args):
        
        # read frame from opencv 
        ret, frame = self.capture.read()

        # cut down frame
        frame = frame[120:120+250,200:200+250, :]

        # flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tostring() # type: ignore
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img
    
    def verify(self, *args):
        # specify thresholds 
        detection_threshold = 0.5
        verification_threshold = 0.5

        # capture input image from our webcam 
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # build results array 
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # make predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        return results, verified
    
if __name__ == '__main__':
    CamApp().run()