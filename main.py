import dlib
from skimage import io
from scipy.spatial import distance
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog as fd, ttk

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
result = 0.0

name = fd.askopenfilename()
print(name)

print("IMAGE", name)
img = io.imread(name)
window1 = dlib.image_window()
window1.clear_overlay()
window1.set_image(img)
dets = detector(img, 1)

for q, t in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        q, t.left(), t.top(), t.right(), t.bottom()))
    shape = sp(img, t)
    window1.clear_overlay()
    window1.add_overlay(t)
    window1.add_overlay(shape)
    face_descriptor1 = facerec.compute_face_descriptor(img, shape)
    print(face_descriptor1)

name2 = fd.askopenfilename()
print(name2)
img2 = io.imread(name2)
window2 = dlib.image_window()
window2.clear_overlay()
window2.set_image(img2)
dets_webcam = detector(img2, 1)
for u, i in enumerate(dets_webcam):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        u, i.left(), i.top(), i.right(), i.bottom()))
    shape = sp(img2, i)
    window2.clear_overlay()
    window2.add_overlay(i)
    window2.add_overlay(shape)
    face_descriptor2 = facerec.compute_face_descriptor(img2, shape)
    print(face_descriptor2)
    a = distance.euclidean(face_descriptor1, face_descriptor2)
    result = a
    print('Result', a)

window = Tk()
window.geometry('400x300')
window.title("Вуриф")
frame = Frame(
    window,
    padx=10,
    pady=10
)
frame.pack(expand=True)
height_lb = Label(
    frame,
    text=result
)
height_lb.grid(row=3, column=1)
height_lb2 = Label(
    frame,
    text="FFF"
)
height_lb2.grid(row=4, column=1)
window.mainloop()

