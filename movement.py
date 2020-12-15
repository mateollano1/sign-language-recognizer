''' This script detects a object of specified object colour from the webcam video feed.
Using OpenCV library for vision tasks and HSV color space for detecting object of given specific color.'''

#--------------------------------------------------------------------------
#------- PLANTILLA DE CÓDIGO ----------------------------------------------
#------- Coceptos básicos de PDI-------------------------------------------
#------- Por: Deiry Sofia Navas Muriel deiry.navas@udea.edu.co --------------
#------- Por: Mateo Llano Avendaño mateo.llano@udea.edu.co --------------
#-------      PFacultad de Ingenieria   -----------------
#------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
#------- Octubre 2020--------------------------------------------------
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#--1. Importación de modulos necesarios en el sistema ---------------------
#--------------------------------------------------------------------------
import cv2
import imutils
import pyautogui
from collections import deque
import time
import numpy as np
import random
# import keras
from tensorflow.python import keras

model = keras.models.load_model('model_train.h5')

class Movement:

#--------------------------------------------------------------------------
#-- 2. Inicialización de variables, carga de imágenes  --------------------
#-------------------------------------------------
# -------------------------
    def main(self):
        #Definr color HSV del rango de color de objetos verdes
        self.greenLower = (29, 86, 6)
        self.greenUpper = (64, 255, 255)

        #Usar en una estructura cola para almancenar los puntos en buffer
        self.buffer = 20

        self.pts = deque(maxlen=self.buffer)
        
        #Inicia video
        self.video_capture = cv2.VideoCapture(0)
        time.sleep(2)


        # Start video capture
        self.video_camera()

    def keras_predict(self, model, image):
        image_data = np.asarray( image, dtype="int32" )
        # result = model.predict_classes(image)
        pred_probab = model.predict(image_data)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        pred_pr, pred_cl = max(pred_probab), pred_class
        print(chr(int(pred_cl)+65))
        word = chr(int(pred_cl)+65)
        # print(word)
        return word

    '''
    Funcion con un ciclo para guardar la captura de la camara infinito
    '''
    def video_camera(self):
        
        while True:
            #---- Almancenar el frame leido -----------------------------------------------
            ret, frame = self.video_capture.read()
            # Dar la vuelta del frame para evitar el efecto de espejo ---------------------
            frame = cv2.flip(frame, 1)
            #---- Cambio de ventana tamaño a 600x600 --------------------------------------
            frame = imutils.resize(frame, width=500)

            x1, y1, x2, y2 = 50, 50, 250, 250
            img_cropped = frame[y1:y2, x1:x2]
            
            #image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            image_grayscale = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
            im3 = cv2.resize(image_grayscale_blurred, (28,28), interpolation = cv2.INTER_AREA)
            im4 = np.resize(im3, (28, 28, 1))
            im5 = np.expand_dims(im4, axis=0)


            cv2.imshow("cropped", img_cropped)
            cv2.imshow("cropped_gray", im4)
            pred_class = self.keras_predict(model, im5)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, pred_class, (130, 300), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0, 0), lineType=cv2.LINE_AA)
            #---- Llamado de la función para aplicar las técnicas -------------------------
            hsv_converted_frame = self.filter_techniques(frame)
           
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF

            #----  Si q se sale de la ventana --------------------------------------------------
            if(key == ord('q')): 
                self.quit()


    def load_image(self, image_url):
        return cv2.imread(image_url)


#--------------------------------------------------------------------------
#-- 4. Tecnicas de filtrado sobre las imágenes  ---------------------------
#--------------------------------------------------------------------------
    def filter_techniques(self,frame):
        #----  Aplicar filtro gaussian bllur de tamaño 5, remover el exceso de ruido -----------
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        #----  Convertir frame rgb a hsv para mejor segmentacion -------------------------------
        hsv_converted_frame = cv2.cvtColor(
            blurred_frame, cv2.COLOR_BGR2HSV)
        return hsv_converted_frame


    def quit(self):
        self.video_capture.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    movement = Movement()
    movement.main()

