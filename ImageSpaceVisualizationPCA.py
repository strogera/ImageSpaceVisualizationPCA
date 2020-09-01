import os.path
import sys
from PIL import Image
import numpy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt


imageMaxSizeX=100
imageMaxSizeY=100
def loadImagesForPCA(path):
    if not os.path.exists(path):
        print("Path Error\nInvalid Path: %s"%path)
        exit()
    imageList=os.listdir(path)
    imgsData=[]
    for image in imageList:
        img=Image.open(path+'/'+image)
        imgResized = img.resize((imageMaxSizeX, imageMaxSizeY), Image.ANTIALIAS)
        #imgOrig.append(img.thumbnail((imageMaxSizeX, imageMaxSizeY), Image.ANTIALIAS))
        imgsData.append(imgResized)
    return imgsData

def PCA_ImageSpaceVisualization(imData, standardize=False):
    x=[numpy.array(img).flatten() for img in imData] #vectorize images
    
    if standardize:
        x=StandardScaler().fit_transform(x)
        
    pca=PCA(n_components=2)
    principalComponents=pca.fit_transform(x)
   
    plt.figure(figsize=(20, 10))
    ax = plt.subplot()

    for i,img in enumerate(principalComponents):
        imagebox=OffsetImage(imData[i], zoom=1)
        imagebox.image.axes=ax
        ax.add_artist(AnnotationBbox(imagebox, principalComponents[i], xybox=(0., 0.), xycoords='data', boxcoords="offset points"))
        
    #dynamically calculate the x,y limits for the plot so the images are displayed in case of stardardize=True
    xmax=max(principalComponents, key=lambda x: x[0])[0]
    xmin=min(principalComponents, key=lambda x: x[0])[0]
    ymax=max(principalComponents, key=lambda x: x[1])[1]
    ymin=min(principalComponents, key=lambda x: x[1])[1]
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv)==2:
        imagesPath=sys.argv[1]
        PCA_ImageSpaceVisualization(loadImagesForPCA(imagesPath))
    else:
        print("Path Error\nUsage: python ImageSpaceVisualizationPCA.py <PathToImagesFolder>")
