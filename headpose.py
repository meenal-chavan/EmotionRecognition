import numpy as np
import cv2
import dlib
from imutils.face_utils import shape_to_np
import face

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0,
     0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# focal_length = size[1]
# center = (size[1]/2, size[0]/2)
# cam_matrix = np.array(
#                         [[focal_length, 0, center[0]],
#                         [0, focal_length, center[1]],
#                         [0, 0, 1]],
#                         dtype = "double"
#                      )
 

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

head_limit = {  "X_upper": 15,
                "X_lower":-10,
                "Y_upper": 15,
                "Y_lower":-15,
                "Z_upper": 15,
                "Z_lower":-15    }

reprojectsrc = np.float32([[0, 0, 0],
                           [10.0, 0, 0],
                           [0, 10.0, 0],
                           [0, 0, 10.0]])

line_pairs = [[0, 1], [0, 2], [0, 3]]


	#hbottom=htop=hleft=hleft_tilt=hright=hright_tilt=hstraight=0

def paint_axes(frame, rotation_vec, translation_vec):

    projected, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec,
                                  cam_matrix, dist_coeffs)
    projected = tuple(map(tuple, projected.reshape(4, 2)))

    for start, end in line_pairs:
        cv2.line(frame, projected[start], projected[end],
                 (0,0,255), thickness=2)

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    ok, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # calculate euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return rotation_vec, translation_vec, euler_angle


def print_head_dir(X,Y,Z,hbottom,htop,hleft,hright,hstraight,hleft_tilt,hright_tilt):
    
    #extern hbottom,htop,hleft,hright,hstraight,hleft_tilt,hright_tilt
    print('HEAD:')
    if X>head_limit["X_upper"]:
        print("bottom")
        hbottom+=1
    if X<head_limit["X_lower"]:
        print("top")
        htop += 1
    if Y>head_limit["Y_upper"]:
        print("right")
        hright+=1
    if Y<head_limit["Y_lower"]:
        print("left")
        hleft+=1
    if Z>head_limit["Z_upper"]:
        print("left tilt")
        hleft_tilt+=1
    if Z<head_limit["Z_lower"]:
        print("right tilt")
        hright_tilt+=1
    return hbottom,htop,hleft,hright,hstraight,hleft_tilt,hright_tilt  
    
def head_straight(X,Y,Z,hstraight):
    if not (X>15 or X<-10 or Y<-15 or Y>15 or Z<-15 or Z>15):
        print("straight")
        hstraight+=1

        return True,hstraight

    return False,hstraight

def extract_coords(euler_angle):
    X = euler_angle[0, 0]
    Y = euler_angle[1, 0]
    Z = euler_angle[2, 0]
    return X, Y, Z

def paint_coords(frame, X,Y,Z, x1=0,y1=0):
    cv2.putText(frame, "X: " + "{:+d}".format(int(round(X))), (x1+20, y1+20), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2)
    cv2.putText(frame, "Y: " + "{:+d}".format(int(round(Y))), (x1+20, y1+50), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2)
    cv2.putText(frame, "Z: " + "{:+d}".format(int(round(Z))), (x1+20, y1+80), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), thickness=2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Unable to initialize Video Source.')
        return
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    i = 1
    while cap.isOpened():

        ok, frame = cap.read()

        if ok:
            i = (i+1)%3

            face_rects = face.detect_faces(frame)

            for rect in face_rects:
                shape = predictor(frame, rect)
                shape = shape_to_np(shape)
                rot_vec, trans_vec, euler_angle = get_head_pose(shape)
                paint_axes(frame, rot_vec, trans_vec)

                X = euler_angle[0, 0]
                Y = euler_angle[1, 0]
                Z = euler_angle[2, 0]
                cv2.putText(frame, "X: " + "{:+d}".format(int(round(X))), (20, 60+20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:+d}".format(int(round(Y))), (20, 60+50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:+d}".format(int(round(Z))), (20, 60+80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)

                # print('----------------------')
                if(i==0): print_head_dir(X,Y,Z,hbottom,htop,hleft,hright,hstraight,hleft_tilt,hright_tilt)

            cv2.imshow('Output', frame)
            key = cv2.waitKey(50)
            if key in [27, ord('q')]:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()