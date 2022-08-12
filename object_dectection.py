import numpy as np
import time
import cv2

labels="synset_words.txt"
model="bvlc_googlenet.caffemodel"
prototxt="bvlc_googlenet.prototxt"
img_path=".//images//vending_machine.png"

image=cv2.imread(img_path)
row=open(labels).read().strip().split("\n")
classes=[r[r.find(" ")+1:].split(",")[0] for r in row]

blob=cv2.dnn.blobFromImage(image,1,(224,224),(104,117,123))
print("Loading model......................")
net=cv2.dnn.readNetFromCaffe(prototxt,model)
net.setInput(blob)
start=time.time()
pred=net.forward()
end=time.time()
print("time took: ",(end-start))
idxs=np.argsort(pred[0])[::-1][:5]
print(idxs)
for (i, idx) in enumerate(idxs):
    if(i==0):
        text = "Label: {}, {:.2f}%".format(classes[idx],
                                           pred[0][idx] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                                    classes[idx], pred[0][idx]))
cv2.imshow("image",image)
cv2.waitKey(0)

