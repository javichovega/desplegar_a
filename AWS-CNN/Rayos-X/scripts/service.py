#####################################
# -*- coding: utf-8 -*-
# Javier Vega Garcia
# Convolucional Neural Network - CNN
#####################################
# REALIZAR PREDICCIONES - Ejercicio_A
#####################################
from flask import Flask, request
from executor import Modelo_RNA
from keras.preprocessing import image
import numpy as np

# Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = Modelo_RNA()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a JVG Enterprises - CNN !!!'

@app.route('/rayos-x/', methods=['GET','POST'])
def rayosx():
	return 'Modelo CNN de Rayos-X !!!'

@app.route('/rayos-x/default/', methods=['GET','POST'])
def default():
	print (request.args)
	img_width, img_height = 299, 299 # dimensions of our images.

	test_image_path = '../samples/' + request.args.get("imagen")
	test_image = image.load_img(test_image_path,	target_size = (img_width, img_height))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0) * 1./255

	with graph.as_default():
		result = loaded_model.predict(test_image)
		resultado = 'Prediccion: ' + ('Abdomen' if result < 0.5 else 'Pulmon') + " X-Ray, su score: " + str(result[0][0])
		return(resultado)

# Run de application
app.run(host='0.0.0.0', port=5000)
