import cv2
import numpy as np

def dashed_rectangle(img, top_left, bottom_right, 
        line_length=5, space_length=2, color=(0, 0, 255), thickness=5):
    # Here is an example
    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img,(0,0),(511,511),(255,0,0),5)

    x_min, y_min = top_left
    x_max, y_max = bottom_right

    step = line_length + space_length

    # from top left to top right, bottom left to bottom right
    for point in range(x_min, x_max, step):
        cv2.line(img, (point, y_min), (point+line_length, y_min), color, thickness)
        cv2.line(img, (point, y_max), (point+line_length, y_max), color, thickness)

    # from TOP to BOTTOM
    for point in range(y_min, y_max, step):
        cv2.line(img, (x_min, point), (x_min, point+line_length), color, thickness)
        cv2.line(img, (x_max, point), (x_max, point+line_length), color, thickness)

if __name__ == '__main__':
    img = np.random.random((256, 256, 3))
    dashed_rectangle(img, (50, 50), (200, 200), line_length=5, space_length=5, thickness=3)
    cv2.imshow('box', img)
    cv2.waitKey(0)
    while True:
        if cv2.waitKey(0)&0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

