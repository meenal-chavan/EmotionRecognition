
import cv2

eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def print_eye_dir(frame,eye_stats):
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        eyes = eyeCascade.detectMultiScale(gray,1.1,5)

        for (x,y,w,h) in eyes:
            roi = frame[y: y+h, x: x+w]
            roi = cv2.resize(roi,(0,0),fx=4,fy=4, interpolation=cv2.INTER_LINEAR)
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5,5), 0)

            _, threshold = cv2.threshold(gray_roi, 25, 255, cv2.THRESH_BINARY_INV)
            contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            ox , oy = rows/2 , cols/2
            # cv2.imshow('Out1', threshold)
            # cv2.imshow('Output2', frame)
            # key = cv2.waitKey(10)

            # print('EYESSSSS')
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)

                midx = x + int(w/2)
                midy = y + int(h/2)
                
                currx,curry=ox-midx,oy-midy

                print('EYE:')
                
                if currx>0:
                    if currx<10 and curry>10:
                        print("top")
                        eye_stats["etop"]+=1
                    elif currx<10 and curry<-10:
                        print("bottom")
                        eye_stats["ebottom"]+=1
                    elif currx>=10:
                        if curry>10:
                            print("top-right")
                            eye_stats["etop_right"]+=1
                        elif curry<-10:
                            print("bottom right")
                            eye_stats["ebottom_right"]+=1
                        elif curry<=10:
                            print("right")
                            eye_stats["eright"]+=1
                    else :
                        print("straight")
                        eye_stats["estraight"]+=1
                
                if currx<0:
                    if currx>-10 and curry>10:
                        print ("top")
                        eye_stats["etop"]+=1
                    if currx>-10 and curry<-10:
                        print("bottom")
                        eye_stats["ebottom"]+=1
                    elif currx<=-10:
                        if curry>10 :
                            print("top-left")
                            eye_stats["etop_left"]+=1
                        elif curry<-10:
                            print("bottom left")
                            eye_stats["ebottom_left"]+=1
                        elif curry<=10:
                            print("left")
                            eye_stats["eleft"]+=1
                    else :
                        print("straight")
                        eye_stats["estraight"]+=1
                    
                return eye_stats
            # break