import os
import cv2
import numpy as np
import scipy
from flask import Flask, redirect, render_template, request, session, abort, url_for

app = Flask(__name__)
APP_ROOT = 'C:/users/Kiki/PycharmProjects/FractalDimProject/'

#Definirame funkcija koja konvertira slika od rgb vo crno-bela, odnosno grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#Presmetka na fraktalnata dimenzija na slikata vo crno-belo
def fractal_dimension(z, threshold=0.9):
    # Only for 2d images
    assert(len(z.shape) == 2)
   
    def boxcount(z, k):
        S = np.add.reduceat(
            np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0),
                               np.arange(0, z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    z = (z < threshold)

    # Minimal dimension of image
    p = min(z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


@app.route('/', methods=['GET','POST'])
def main():
    return render_template("upload.html")


@app.route('/upload', methods=['POST'])
def upload():
    #print(APP_ROOT)
    target = os.path.join(APP_ROOT, 'static')
    #print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {} ".format(target))
    print(request.files.getlist("file"))
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        img=cv2.imread(destination)
        #cv2.imshow('image', img)
        image = rgb2gray(img)
        dim = fractal_dimension(image)
        print(dim)


    #print("Theoretical Hausdorff dimension:", (np.log(3) / np.log(2)))
    return render_template('upload.html', image_name=filename,output=dim)


if __name__ == '__main__':
    app.run()