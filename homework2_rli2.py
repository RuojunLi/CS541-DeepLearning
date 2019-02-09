# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np

def J (w, faces, labels, alpha):
    e = np.dot(faces,w) - labels
    l2 = alpha * np.dot(w.T,w)
    return 0.5 * (np.dot(e.T,e) + l2)  # TODO fix this!

def gradJ (w, faces, labels, alpha):
    return np.dot(faces.T,np.dot(faces,w) - labels) + (alpha * w)  # TODO fix this!

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.zeros(trainingFaces.shape[1])  # Or set to random vector
    epsilon = 2e-6 # 0.000001
    delta = 1e-4 # 0.0001
    prevJ = J(w, trainingFaces, trainingLabels, alpha)
    diff = 1000
    count = 0
    # print(diff, delta)
    while diff > delta:
        update = epsilon * gradJ(w, trainingFaces, trainingLabels, alpha)
        w = w - update
        newJ = J(w, trainingFaces, trainingLabels, alpha)
        diff = prevJ - newJ
        prevJ = newJ
        count += 1
        if count % 1000 == 0: print('\tcost:',newJ,'\tDiff:',diff)
    return w

def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha):
    w = np.zeros(trainingFaces.shape[1])  # TODO fix this!
    left = np.dot(trainingFaces.T, trainingFaces)
    right = np.dot(trainingFaces.T, trainingLabels)
    w = np.linalg.solve(left, right)
    return w

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)

def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print("Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha)))
    print("Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha)))

# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles (w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile (im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1]+faceBox[3], faceBox[0]:faceBox[0]+faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0]*face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        print(yhat)

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0]+faceBox[2], faceBox[1]+faceBox[3])
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)

    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # TODO update the path
    while vc.grab():
        (tf,im) = vc.read()
        im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))  # Divide resolution by 2 for speed
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print("quitting")
            break

        # Detect faces
        faceBoxes = faceDetector.detectMultiScale(imGray)
        for faceBox in faceBoxes:
            classifySmile(im, imGray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

    ALPHA = 1E3
    w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels, ALPHA)
    w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels, ALPHA)

    for w in [ w1, w2 ]:
        reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
    
    #detectSmiles(w3)  # Requires OpenCV
