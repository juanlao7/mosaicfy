import argparse
import sys
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel, sobel_h, sobel_v

MEASURE_MIN_TIME = False

def showImage(image):
    plt.imshow(image)
    plt.show()

def changeBrightness(color, proportion):
    newColor = color * proportion
    m = newColor.max()

    if m > 255:
        newColor *= 255 / m

    color[:] = newColor[:]

def makeItSmooth(image, mask, newRows, newColumns, axis, isReversed):
    if axis == 1:
        image = np.swapaxes(image, 0, 1)
        mask = np.swapaxes(mask, 0, 1)

    for x in range(image.shape[0]):
        iterator = range(image.shape[1])

        if isReversed:
            iterator = reversed(iterator)
        
        lastY = None

        for y in iterator:
            if mask[x, y]:
                if lastY is not None and ((x > 0 and mask[x - 1, lastY]) or (x < image.shape[0] - 1 and mask[x + 1, lastY])):
                    image[x, lastY] = (0.75, 0.75, 0.75)

                    if axis == 0:
                        newRows.append(x)
                        newColumns.append(lastY)
                    else:
                        newRows.append(lastY)
                        newColumns.append(x)

                break
            
            lastY = y

def getSmoothMask(mask):
    smoothMask = np.dstack((mask, mask, mask)).astype(float)
    return smoothMask

    # The following lines apply an antialias, but it's too slow.
    newRows = []
    newColumns = []
    makeItSmooth(smoothMask, mask, newRows, newColumns, 0, False)
    makeItSmooth(smoothMask, mask, newRows, newColumns, 0, True)
    makeItSmooth(smoothMask, mask, newRows, newColumns, 1, False)
    makeItSmooth(smoothMask, mask, newRows, newColumns, 1, True)
    mask[newRows, newColumns] = True
    return smoothMask

def erode(mask, axis, isReversed):
    if axis == 1:
        mask = np.swapaxes(mask, 0, 1)
    
    for x in range(mask.shape[0]):
        if random.getrandbits(1):
            iterator = range(mask.shape[1])

            if isReversed:
                iterator = reversed(iterator)
        
            for y in iterator:
                if mask[x, y]:
                    mask[x, y] = False
                    break

def putTile(image, result, mask, maskBounds, brightnessCorrection, randomBrightnessChange):
    erode(mask, 0, False)
    erode(mask, 1, False)
    erode(mask, 0, True)
    erode(mask, 1, True)

    maskXs, maskYs = np.where(mask)
    color = getAverageColor(image, maskXs, maskYs, maskBounds)

    if color is None:
        return
    
    changeBrightness(color, brightnessCorrection * (1 + (random.random() - 0.5) * randomBrightnessChange))
    result[maskBounds[0]:maskBounds[2] + 1, maskBounds[1]:maskBounds[3] + 1][maskXs, maskYs] = getSmoothMask(mask)[maskXs, maskYs] * color

def getAverageColor(image, maskXs, maskYs, maskBounds):
    if maskXs.size == 0:
        return None

    pixels = image[maskBounds[0]:maskBounds[2] + 1, maskBounds[1]:maskBounds[3] + 1][maskXs, maskYs]
    return pixels.mean(axis=0)
    #return np.sqrt((pixels.astype(float) ** 2).mean(axis=0)).astype(np.uint8)      # This is the correct way of computing the mean, but it's slower.

def createRectangleBounds(w, h, tileSize, randomVariation):
    randomVariation = int(randomVariation * tileSize)
    minTileSize = tileSize - randomVariation
    maxTileSize = tileSize + randomVariation
    w -= 2
    h -= 2

    x1 = 1
    x2 = x1 + random.randint(minTileSize, maxTileSize)
    
    while x2 < w:
        y1 = 1
        y2 = y1 + random.randint(minTileSize, maxTileSize)

        while y2 < h:
            yield (x1, y1, x2, y2)
            y1 = y2 + 2
            y2 = y1 + random.randint(minTileSize, maxTileSize)
        
        if y1 < h:
            yield (x1, y1, x2, h)

        x1 = x2 + 2
        x2 = x1 + random.randint(minTileSize, maxTileSize)
    
    if x1 < w:
        y1 = 1
        y2 = y1 + random.randint(minTileSize, maxTileSize)

        while y2 < h:
            yield (x1, y1, w, y2)
            y1 = y2 + 2
            y2 = y1 + random.randint(minTileSize, maxTileSize)
        
        if y1 < h:
            yield (x1, y1, w, h)

def process(inputImage, horizontalDivisions, randomVariation, gradientThreshold, randomBrightnessChange):
    start = time.time()

    inputImageGray = rgb2gray(inputImage)
    horizontalSobel = sobel_h(inputImageGray)
    verticalSobel = sobel_v(inputImageGray)
    fullSobel = sobel(inputImageGray)

    result = np.zeros(inputImage.shape, dtype=np.uint8)
    tileSize = inputImage.shape[0] // horizontalDivisions
    brightnessCorrection = 1 / (((tileSize - 1) ** 2) / (tileSize ** 2))

    for bounds in createRectangleBounds(inputImage.shape[0], inputImage.shape[1], tileSize, randomVariation):
        shape = (bounds[2] - bounds[0] + 1, bounds[3] - bounds[1] + 1)
        c = np.unravel_index(np.argmax(fullSobel[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1]), shape)
        
        if fullSobel[bounds[0] + c[0], bounds[1] + c[1]] < gradientThreshold:
            putTile(inputImage, result, np.ones(shape, dtype=bool), bounds, brightnessCorrection, randomBrightnessChange)
            continue
        
        horizontalSum = horizontalSobel[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1].sum()
        verticalSum = verticalSobel[bounds[0]:bounds[2] + 1, bounds[1]:bounds[3] + 1].sum()
        d = (c[0] - verticalSum, c[1] + horizontalSum)
        xy = np.array(list(np.ndindex(shape)))
        mask1 = ((c[1] - xy[:, 1]) * (d[0] - c[0]) - (c[0] - xy[:, 0]) * (d[1] - c[1]) > 0).reshape(shape)        # Orientation of point (x, y) according to vector <d, c>
        mask2 = np.invert(mask1)
        putTile(inputImage, result, mask1, bounds, brightnessCorrection, randomBrightnessChange)
        putTile(inputImage, result, mask2, bounds, brightnessCorrection, randomBrightnessChange)

    return result, time.time() - start

parser = argparse.ArgumentParser(description='Convert any image into a mosaic.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input_image', help='File path of the input image. Compatible with most common image formats.')
parser.add_argument('output_image', help='File path of the output image. Format is inferred from the extension of the file name. Compatible with most common image formats.')
parser.add_argument('-d', dest='horizontal_divisions', default=100, type=int, help='Approximate number of tiles in each row.')
parser.add_argument('-v', dest='random_variation', default=0.25, type=float, help='Random proportional variation applied to the size of each tile.')
parser.add_argument('-g', dest='gradient_threshold', default=0.1, type=float, help='Used for detecting borders and splitting tiles to match them. Between 0 (all tiles must be splitted) and 1 (no tiles must be splitted).')
parser.add_argument('-b', dest='random_brightness_change', default=0.75, type=float, help='Range of the random brightness variation applied to each tile.')
parser.add_argument('-r', dest='random_seed', default=0, type=int, help='Random seed.')
args = parser.parse_args()

random.seed(args.random_seed)
inputImage = plt.imread(args.input_image)

if inputImage.ndim < 3:
    sys.exit('Only RGB or RGBA images supported.')
elif inputImage.ndim == 4:
    inputImage = inputImage[:, :, :3]

if MEASURE_MIN_TIME:
    times = np.zeros(5)

    for i in range(5):
        _, duration = process(inputImage, args.horizontal_divisions, args.random_variation, args.gradient_threshold, args.random_brightness_change)
        times[i] = duration

    print(times)
    print(times.min())
else:
    result, duration = process(inputImage, args.horizontal_divisions, args.random_variation, args.gradient_threshold, args.random_brightness_change)
    print('Processed in', round(duration, 2), 'seconds.')
    plt.imsave(args.output_image, result)
