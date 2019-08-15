import pygame
from pygame.locals import *
import sys
from PIL import Image
import numpy as np
import os

#---------------------User set---------------------
SCREEN_SIZE = [320,400]
BAR_SIZE = [60, 5]
BALL_SIZE = [15, 15]
#training image size_x
imagesize_x = 50
#training image size_y
imagesize_y = 100
# Set windows position
xPos = 580
yPos = 405
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (xPos,yPos)
#--------------------------------------------------
BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)
 
class Game(object):
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption('Simple Game')
		self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
		self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
		self.ball_dir_x = -1 # -1 = left 1 = right  
		self.ball_dir_y = -1 # -1 = up   1 = down
		self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
		self.score = 0
		self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
		self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
	def run(self):
		pygame.mouse.set_visible(0) # make cursor invisible
		while(1):
			#----------------------------keyboard event-----------------------------
			for event in pygame.event.get():#Close
				if event.type == QUIT:
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
			#-----------------------------shoot screen------------------------------
			im = pygame.surfarray.array3d(self.screen)
			im = Image.fromarray(im, 'RGB')#array to image
			im = im.resize((imagesize_x,imagesize_y))
			im = im.convert("L")#RGB to Gray	
			im_buffer = np.array(im)#image to array
			im.close()
			im_buffer = im_buffer.reshape(1,imagesize_x,imagesize_y,1)
			#----------------------------Action predict-----------------------------
			action = model.predict_classes(im_buffer)[0]
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
			pygame.draw.rect(self.screen, WHITE, self.bar_pos)
 
			self.ball_pos.left += self.ball_dir_x * 2
			self.ball_pos.bottom += self.ball_dir_y * 3
			pygame.draw.rect(self.screen, WHITE, self.ball_pos)
 
			if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
				self.score = self.score + 1
				sys.stdout.write("\rScore %s" %(self.score))
				sys.stdout.flush()
				if self.score >= 500:#Redom ball_position test
					print("")
					print("\rRestart Game: ",self.score)
					self.restart()
					pygame.draw.rect(self.screen, WHITE, self.ball_pos)#Upgrade screen
					pygame.time.wait(600)
			elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
				print("\rGame Over: ", self.score)
				self.restart()
				pygame.draw.rect(self.screen, WHITE, self.ball_pos)
				pygame.time.wait(600)
				
			if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]+1):
				self.ball_dir_y = self.ball_dir_y * -1
			if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
				self.ball_dir_x = self.ball_dir_x * -1
				
			pygame.display.update()
			self.clock.tick(60)#Max fps
		print("")
	def restart(self):
		self.ball_pos.left = np.random.randint(1,self.ball_pos_range[0])#can't touch screen_left
		self.ball_pos.top = np.random.randint(1,self.ball_pos_range[1])#can't touch screen_top
		self.ball_dir_x,self.ball_dir_y = np.random.choice([1,-1], size=2, replace=True)
		self.score = 0

import bulid_model
#------------------------Use GPU (ERROR Blas GEMM launch failed)-----------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
session = tf.Session(config=config)
#--------------------------------------------------------------------------
model = bulid_model.bulid_model(3,imagesize_x,imagesize_y)#bulid_model
model.load_weights("model.weight")

game = Game()
game.run()


