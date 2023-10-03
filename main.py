import tensorflow as tf
tf.get_logger().setLevel('INFO')
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import threading
from time import sleep
import cv2

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
from threading import Thread

class teachablemachine():

	frame=None
	class_name=""
	confidence_score=0

	def __init__(self):

		cam_loop_thread = threading.Thread(target=self.cam_loop)
		cam_loop_thread.start()

		sleep(2)
		self.initAI()
		AI_loop_thread=threading.Thread(target=self.AI_loop)
		AI_loop_thread.start()


	def initAI(self):
		# Disable scientific notation for clarity
		np.set_printoptions(suppress=True)

		# Load the model
		self.model = load_model("keras_model.h5", compile=False)

		# Load the labels
		with open("labels.txt", 'r') as myfile:
			self.class_names = [line.rstrip() for line in myfile.readlines()]
		
		#print(self.class_names)
		
  
	def cam_loop(self):
		
		# CAMERA can be 0 or 1 based on default camera of your computer
		self.camera = cv2.VideoCapture(0)

		while self.camera.isOpened():
			# Grab the webcamera's image.
			ret, frame = self.camera.read()

			# Resize the raw image into (224-height,224-width) pixels
			self.frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

			# Show the image in a window
			cv2.imshow("Webcam Image",frame)
				
			# Listen to the keyboard for presses.
			keyboard_input = cv2.waitKey(1)
		
			# 27 is the ASCII for the esc key on your keyboard.
			if keyboard_input == 27:
				break
				
		self.camera.release()

	def AI_loop(self):
		while True:
			if self.frame is not None:
				# Replace this with the path to your image
				#image = Image.open("<IMAGE_PATH>").convert("RGB")
				image=self.frame.copy()
				image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
				# resizing the image to be at least 224x224 and then cropping from the center
				
				# Make the image a numpy array and reshape it to the models input shape.
				image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

				# Normalize the image array
				image = (image / 127.5) - 1

				# Predicts the model
				prediction = self.model.predict(image)
				index = np.argmax(prediction)
				self.class_name = self.class_names[index]
				self.confidence_score = prediction[0][index]

				# Print prediction and confidence score
				#print("Class:", self.class_name,self.confidence_score)
				#print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")


MACHINE=teachablemachine()


####################################################################

robot = Create3(Bluetooth('MonicaRoomba'))

@event(robot.when_play)
async def play(robot):
	# Trigger an undock and then dock. Try putting this in an infinite loop!
	print('Undock')
	print(await robot.undock())
	

@event(robot.when_play)
async def play(robot):
	# Dock sensor visualizer; could be improved with events
	
	action="iddle"

	while True:
		left=0
		right=0
		speed=2
	   
		#sensor = (await robot.get_docking_values())['IR sensor 0']
		r = 255 * ((MACHINE.confidence_score & 8)/8)
		g = 255 * ((MACHINE.confidence_score & 4)/4)
		b = 255 * (MACHINE.confidence_score & 1)
		await robot.set_lights_rgb(r, g, b)
		
		#choose action
		if MACHINE.confidence_score>0.8:
			action=MACHINE.class_name

		print(action)

		#apply action
		if action=="left":
			right=speed
			left=-speed

		if action=="right":
			right=-speed
			left=speed

		if action=="forward":
			right=speed
			left=speed

		if action=="backward":
			right=-speed
			left=-speed

		
		#do move the wheels
		await robot.set_wheel_speeds(left,right)

robot.play()
