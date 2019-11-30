import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import face

EAR_THRESH = 0.22
# EYE_AR_CONSEC_FRAMES = 10



def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = distance.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

def eye_is_closed(shape):
    (leftEye_s, leftEye_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rightEye_s, rightEye_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye  = shape[leftEye_s:leftEye_e]
    rightEye = shape[rightEye_s:rightEye_e]

    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

    isclosed = ear < EAR_THRESH
    return isclosed


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Unable to initialize Video Source.')
        return
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    COUNTER = BLINK_COUNT = 0
    while cap.isOpened():

        ok, frame = cap.read()

        if ok:

            face_rects = face.detect_faces(frame)

            for rect in face_rects:
                shape = predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)

                if eye_is_closed(shape):
                    COUNTER += 1
                else:
                    if COUNTER is not 0:
                        COUNTER = 0
                        BLINK_COUNT += 1
                        print('Blinked')

            print(BLINK_COUNT)
            cv2.imshow('Output', frame)
            key = cv2.waitKey(50)
            if key in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()