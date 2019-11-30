import cv2
import dlib

detector = dlib.get_frontal_face_detector()

def detect_faces(frame):
    face_rects = detector(frame, 0)
    return face_rects

def get_rect_pos(rect):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return x1, y1, x2, y2

def paint_rectangle(frame, rect):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 2)

def detect_faces_and_paint(frame):
    for rect in detect_faces(frame):
        paint_rectangle(frame, rect)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Unable to initialize Video Source.')
        return
    
    while cap.isOpened():

        ok, frame = cap.read()

        if ok:

            detect_faces_and_paint(frame)
            cv2.imshow('Output', frame)
            key = cv2.waitKey(10)
            if key in [27, ord('q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()