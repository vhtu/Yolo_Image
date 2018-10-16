import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

tfnet = TFNet(options)

img = cv2.imread('p3.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)

print (result)
print (img.shape)

# pull out some info from the results# pull o 

print ("So doi tuong trong hinh: "+str(len(result)))
for x in result:
	tl = (x['topleft']['x'], x['topleft']['y'])
	br = (x['bottomright']['x'], x['bottomright']['y'])
	label = x['label']


	# add the box and label and display it
	img = cv2.rectangle(img, tl, br, (0, 255, 0), 2)
	img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


plt.imshow(img)
plt.show()
