import cv2
import numpy as np

img_rgb = cv2.imread('dataset/test.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = []
template.append(cv2.imread('dataset/template_h.png', 0))
template.append(cv2.imread('dataset/template_d.png', 0))
template.append(cv2.imread('dataset/template_c.png', 0))
template.append(cv2.imread('dataset/template_s.png', 0))

i = 0
sign = ['hearts', 'diamond', 'clubs', 'spades']

for temp in template:
    w, h = temp.shape[::-1]
    res = cv2.matchTemplate(img_gray, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    if loc[0].size > 0:
        print('Sign detected : ' + sign[i])
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 0)
        cv2.imshow('Detected', img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    i += 1

