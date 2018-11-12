#####################################
# -*- coding: utf-8 -*-
# Javier Vega Garcia
# Convolucional Neural Network - CNN
#####################################
# ACTIVAR EL SERVICIO - Ejercicio_A
#####################################
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam

def Modelo_RNA():
	print("Cargando modelo desde el disco ...")
	file_model = "../model/rx_model_jvg.json"
	file_pesos = "../model/rx_model_jvg.h5"

	json_file = open(file_model, 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(file_pesos)
	loaded_model.compile(optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss = 'binary_crossentropy', metrics = ['accuracy'])
	print("Modelo cargado !!!")
	graph = tf.get_default_graph()
	return loaded_model, graph
