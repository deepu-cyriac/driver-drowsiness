import cv2,urllib.request
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
from django.conf import settings
from twilio.rest import Client
import time
from scipy.spatial import distance as dist



class VideoCamera(object):
	#eye detection variables
	thresh = 0.25
	frame_check = 20
	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor("/home/deepu/Main/main_a/m_cam/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	flag1=0

	#yawning variables
	yawn_thresh = 20
	ptime = 0
	count = 0
	yawn = False
	#-------Models---------#
	face_model = dlib.get_frontal_face_detector()
	landmark_model = dlib.shape_predictor("/home/deepu/Main/main_a/m_cam/shape_predictor_68_face_landmarks.dat")
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def cal_yawn(self, shape):
		top_lip = shape[50:53]
		top_lip = np.concatenate((top_lip, shape[61:64]))

		low_lip = shape[56:59]
		low_lip = np.concatenate((low_lip, shape[65:68]))

		top_mean = np.mean(top_lip, axis=0)
		low_mean = np.mean(low_lip, axis=0)

		distance = dist.euclidean(top_mean,low_mean)
		return distance

	def eye_aspect_ratio(self,eye):
		A = distance.euclidean(eye[1], eye[5])
		B = distance.euclidean(eye[2], eye[4])
		C = distance.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)
		return ear

	def get_frame(self):
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		ret, frame = self.video.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = self.detect(gray, 0)

		for subject in subjects:
			shape = self.predict(gray, subject)
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array
			leftEye = shape[self.lStart:self.lEnd]
			rightEye = shape[self.rStart:self.rEnd]
			leftEAR = self.eye_aspect_ratio(leftEye)
			rightEAR = self.eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			if ear < self.thresh:
				self.flag1+= 1
				#print (self.flag1)
				if self.flag1 >= self.frame_check:
					cv2.putText(frame, "Alert", (200, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					cv2.putText(frame, "Drowsiness Detected", (150,325),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
			else:
				self.flag1 = 0
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_flip = cv2.flip(frame,1)
		ret1, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()

class IPWebCam(object):
	thresh = 0.25
	frame_check = 20
	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor("/home/deepu/Main/main_a/m_cam/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	flag2=0

	#yawning variables
	yawn_thresh = 20
	count = 0
	yawn = False
	#-------Models---------#
	face_model = dlib.get_frontal_face_detector()
	landmark_model = dlib.shape_predictor("/home/deepu/Main/main_a/m_cam/shape_predictor_68_face_landmarks.dat")

	def cal_yawn(self, shape):
		top_lip = shape[50:53]
		top_lip = np.concatenate((top_lip, shape[61:64]))

		low_lip = shape[56:59]
		low_lip = np.concatenate((low_lip, shape[65:68]))

		top_mean = np.mean(top_lip, axis=0)
		low_mean = np.mean(low_lip, axis=0)

		distance = dist.euclidean(top_mean,low_mean)
		return distance

	def twilcall(self):
		client = Client()
		call = client.calls.create(
			from_='+19704898161',
			to='+918137083598',
			url='https://handler.twilio.com/twiml/EH8ad2d055828cf50c59a225a06cc5e998'
		)

	def eye_aspect_ratio(self,eye):
		A = distance.euclidean(eye[1], eye[5])
		B = distance.euclidean(eye[2], eye[4])
		C = distance.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)
		return ear

	def __init__(self):
		self.url = "http://192.168.1.3:8080/shot.jpg?rnd=552582"

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		imgResp = urllib.request.urlopen(self.url)
		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
		frame= cv2.imdecode(imgNp,-1)

		
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = self.detect(gray, 0)
		for subject in subjects:
			shape = self.predict(gray, subject)
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array

			#yawn
			#-------Detecting/Marking the lower and upper lip--------#
			lip = shape[48:60]
			cv2.drawContours(frame,[lip],-1,(0, 0, 255),thickness=3)

			#-------Calculating the lip distance-----#
			lip_dist = self.cal_yawn(shape)
			# print(lip_dist)
			if lip_dist > self.yawn_thresh :
				self.count += 1
				if (self.count < 4):
					cv2.putText(frame, f'User Yawning!',(frame.shape[1]//2 - 170 ,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2)
					print(self.count)
				else:
					cv2.putText(frame, f'Drowsiness Detected!',(frame.shape[1]//2 - 170 ,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2)
					self.twilcall()
					self.count = 0
			leftEye = shape[self.lStart:self.lEnd]
			rightEye = shape[self.rStart:self.rEnd]
			leftEAR = self.eye_aspect_ratio(leftEye)
			rightEAR = self.eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			if ear < self.thresh:
				self.flag2 += 1
				print (self.flag2)
				if self.flag2 >= self.frame_check:
					self.twilcall()
					self.flag2=0
			else:
				self.flag2 = 0
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		resize = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR) 
		frame_flip = cv2.flip(resize,1)
		ret, jpeg = cv2.imencode('.jpg', frame_flip)
		return jpeg.tobytes()