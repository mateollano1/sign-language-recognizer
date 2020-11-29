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
from movement import Movement
import threading

class Main:


#--------------------------------------------------------------------------
#--1. Llamado asíncrono de aplicativo de cámara y juego de carros----------
#--------------------------------------------------------------------------
    def game_init(self):
        
        movement = Movement()
        movement.main()
        return


if __name__ == '__main__':
    game = Main()
    game.game_init()