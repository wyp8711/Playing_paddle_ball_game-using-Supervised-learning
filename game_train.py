import pygame
from pygame.locals import *
import sys
import numpy
from PIL import Image
import numpy as np
from keras.utils import np_utils
import gc
import os
import csv,time

#---------------------User set---------------------
SCREEN_SIZE = [320,400]
BAR_SIZE = [60, 5]
BALL_SIZE = [15, 15]
#training image size_x
imagesize_x = 50
#training image size_y
imagesize_y = 100
#if you have Unfinished model.weight, you can set False
new_work = True
# Set windows position
xPos = 580
yPos = 405
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (xPos,yPos)
#--------------------------------------------------
BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)
with open('history.csv','w',newline='') as f:
	f = csv.writer(f)
	f.writerow(["Average score","Time of data collection (s)","Time of training (s)"])

class Game(object):
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption('Simple Game')
		self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
		self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
		#1.y can't < bar, 2.Consider ball y_size, 3.bar limit move
		self.ball_pos_range = [SCREEN_SIZE[0]-BALL_SIZE[0],SCREEN_SIZE[1]-(BALL_SIZE[1]+BAR_SIZE[1]+SCREEN_SIZE[0])+BAR_SIZE[0]]
		self.ball_dir_x = -1 # -1 = left 1 = right  
		self.ball_dir_y = -1 # -1 = up   1 = down
		self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
		self.score = 0
		self.mean_score = np.array([])
		self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
		self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
	def run(self,new_work):
		pygame.mouse.set_visible(0) # make cursor invisible
		clock = time.time()#Record start time
		run_time = 0
		save_time = 0
		flag = False
		new_work = new_work
		game_over_flag = True
		test_pass_cnt = 0
		while(1):
			run_time += 1
			#----------------------------keyboard event-----------------------------
			for event in pygame.event.get():
				if event.type == QUIT:#Close
					pygame.quit()
					sys.exit()
				elif event.type == pygame.KEYDOWN and event.key == K_ESCAPE:#Esc
					pygame.quit()
					sys.exit()
				elif event.type == pygame.KEYUP and event.key == K_s:#s
					#pygame.image.save(self.screen, 'fileout.png')
					im = pygame.surfarray.array3d(self.screen)
					im = Image.fromarray(im, 'RGB')
					im.save( "fileout.png" )
				elif event.type == pygame.KEYUP and event.key == K_t:
					sh = Image.fromarray(data_x_bu[2],'L')
					sh.save( "fileout.png" )
			#----------------------------label generator----------------------------
			#bar action : (0:pass, 1:Right, 2:Left)
			bar_center = ((self.bar_pos.right - self.bar_pos.left)/2)+self.bar_pos.left
			ball_center = ((self.ball_pos.right - self.ball_pos.left)/2)+self.ball_pos.left
			if (bar_center < ball_center):
				label = 1
				if self.bar_pos_x >= SCREEN_SIZE[0] - BAR_SIZE[0] :
					label = 0
			elif (bar_center > ball_center):
				label = 2
				if self.bar_pos_x <= 0 :
					label = 0
			else:
				label = 0
			#-----------------------------shoot screen------------------------------
			im = pygame.surfarray.array3d(self.screen)
			im = Image.fromarray(im, 'RGB')#array to image
			im = im.resize((imagesize_x,imagesize_y))
			im = im.convert("L")#RGB to Gray	
			im_buffer = np.array(im)#image to array
			im.close()
			im_buffer = im_buffer.reshape(im_buffer.size)#Flatten(can be easy for save)
			#----------------------------image data stack---------------------------
			if run_time == 1:
				data_x_bu = im_buffer
				data_y_bu = label
			else:
				data_x_bu = np.vstack((data_x_bu,im_buffer))
				data_y_bu = np.hstack((data_y_bu,label))
			#When game_over in this 1500 run_time, Leave(stack) buffer data
			if (run_time >= 1500):
				if game_over_flag:
					if save_time==0:
						data_x = data_x_bu
						data_y = data_y_bu
					else:
						data_x = np.vstack((data_x,data_x_bu))
						data_y = np.hstack((data_y,data_y_bu))
					save_time += 1
					run_time = 0
					del data_x_bu,data_y_bu
					gc.collect()
					game_over_flag = False
				else:
					data_x_bu = data_x_bu[750:]
					data_y_bu = data_y_bu[750:]
					run_time -= 750 
			#---------------------When data > 1500*7, Start train---------------------
			if save_time >= 7:
				print("")#end score print
				data_y = np_utils.to_categorical(data_y, num_classes=3)#decoder(0->100, 1->010, 2->001)
				data_x = data_x.reshape(len(data_x), imagesize_x,imagesize_y,1).astype('float32')#reshape to (samples,size_x,size_y,color_ch)
				data_x = data_x/255 
				#--------History-----------
				#Record mean score & time of data collection
				if new_work == False:#Record time of start fit
					write_data = [round(np.mean(self.mean_score),2),round((time.time()-clock),2)]
				else:#First run, score isn't important
					write_data = ["Create random train data",round((time.time()-clock),2)]
					new_work = False#disable first run flag
				clock = time.time()#Record time of start fit
				#--------Start Fit-----------
				model.fit(x=data_x, y=data_y, validation_split=0.2, epochs=50, batch_size=200, 
												  verbose=2,callbacks = [learning_rate_function,early_stopping])
				write_data.append(time.time()-clock)#Record time of end fit
				model.save_weights("model.weight")
				#---------History------------
				with open('history.csv','a+',newline='') as f:
					f = csv.writer(f)
					f.writerow(write_data)
				#---------restart------------
				self.restart()
				pygame.draw.rect(self.screen, WHITE, self.ball_pos)#Upgrade screen
				#----------------------------
				save_time = 0
				del data_x,data_y,self.mean_score,write_data
				gc.collect()
				self.mean_score = np.array([])
				clock = time.time()#Record time of start data collection
			#-----------------------------------------------------------------------
			#----------------------------action predict-----------------------------
			im_buffer = im_buffer.reshape(1, imagesize_x,imagesize_y,1).astype('float32')#reshape to (samples,size_x,size_y,color_ch)
			action = model.predict_classes(im_buffer)[0]#predict_classes
			#----------------------------Random Controll----------------------------
			#When first run
			if new_work:
				if flag == True:
					if self.bar_pos_x >= SCREEN_SIZE[0] - BAR_SIZE[0]:
						flag = False
				elif self.bar_pos_x > 0:
					flag = False
				else:
					flag = True
				
				if flag == 1:
					action = numpy.random.choice([0,1,2], 1, p=[0.35, 0.45, 0.2])[0]
				else:
					action = numpy.random.choice([0,1,2], 1, p=[0.35, 0.2, 0.45])[0]
			#----------------------------Action Controll----------------------------
			if action == 1:#Right
				self.bar_pos_x = self.bar_pos_x + 2
				if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
				   self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
			elif action == 2:#left
				self.bar_pos_x = self.bar_pos_x - 2
				if self.bar_pos_x < 0:
				   self.bar_pos_x = 0
			else:
				pass
			#----------------------------Screen Process-----------------------------
			self.screen.fill(BLACK)
			self.bar_pos.left = self.bar_pos_x
			pygame.draw.rect(self.screen, WHITE, self.bar_pos)#Upgrade screen
 
			self.ball_pos.left += self.ball_dir_x * 2
			self.ball_pos.bottom += self.ball_dir_y * 3
			pygame.draw.rect(self.screen, WHITE, self.ball_pos)#Upgrade screen
		
			if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
				self.score = self.score + 1
				sys.stdout.write("\rScore %s" %(self.score))
				sys.stdout.flush()
				if self.score >= 500:#Redom ball_position test
					print("")
					print("\rRestart Game: ",self.score)
					test_pass_cnt = test_pass_cnt + 1
					self.mean_score = np.hstack((self.mean_score, self.score))
					self.restart()
					pygame.draw.rect(self.screen, WHITE, self.ball_pos)#Upgrade screen
					pygame.time.wait(600)
					if test_pass_cnt >= 10:
						print("")
						print("Finish...")
						write_data = [round(np.mean(self.mean_score),2),round((time.time()-clock),2),"Finish"]
						with open('history.csv','a+',newline='') as f:
							f = csv.writer(f)
							f.writerow(write_data)
						pygame.quit()
						sys.exit()
			elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
				print("\rGame Over: ", self.score)
				self.mean_score = np.hstack((self.mean_score, self.score))
				self.restart()
				pygame.draw.rect(self.screen, WHITE, self.ball_pos)#Upgrade screen
				test_pass_cnt = 0
				game_over_flag = True
				pygame.time.wait(600)
				
			if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]):
				self.ball_dir_y = self.ball_dir_y * -1
			if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
				self.ball_dir_x = self.ball_dir_x * -1
			
			pygame.display.update()
			self.clock.tick(60)#Max fps
	def restart(self):
		self.ball_pos.left = np.random.randint(1,self.ball_pos_range[0])#can't touch screen_left
		self.ball_pos.top = np.random.randint(1,self.ball_pos_range[1])#can't touch screen_top
		self.ball_dir_x,self.ball_dir_y = np.random.choice([1,-1], size=2, replace=True)
		self.score = 0

#------------------------------------------------------------------------------
import bulid_model
#------------------------Use GPU (ERROR Blas GEMM launch failed)-----------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
session = tf.Session(config=config)
#--------------------------------------------------------------------------
model = bulid_model.bulid_model(3,imagesize_x,imagesize_y)#bulid_model
if not new_work:
	model.load_weights("model.weight")

from keras.callbacks import ReduceLROnPlateau,EarlyStopping
learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

early_stopping = EarlyStopping(monitor='acc',min_delta=0.01, patience=7, verbose=1, mode='max')


game = Game()
game.run(new_work)
