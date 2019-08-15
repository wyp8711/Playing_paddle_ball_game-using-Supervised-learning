from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D,MaxPool2D,BatchNormalization
from keras.optimizers import SGD
#--------------------------------------display model--------------------
#import os
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model#Save model.png
#os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
#-----------------------------------------------------------------------
def bulid_model(label,imagesize_x,imagesize_y):
	model = Sequential()
	model.add(Conv2D(9, (3, 3), activation='relu',data_format = 'channels_last',
						 kernel_initializer='he_uniform', input_shape=(imagesize_x, imagesize_y, 1)))
	model.add(Activation('relu'))	
	model.add(BatchNormalization(axis=3))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(600, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(600,activation='relu'))
	model.add(Dropout(0.25))
	# Add output layer
	model.add(Dense(units=label, kernel_initializer='normal', activation='softmax'))
	opt = SGD(lr=0.01, momentum=0.9)
	
	model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy']) 
	#-------------------display model-------------------
	model.summary()
	#display(SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')))#Show model in message box
	#os.system("del Z:\model.png /f /q" )#Delete old model.png
	#plot_model(model, show_shapes=True, to_file='Z:\model.png')#Save model.png
	return model
