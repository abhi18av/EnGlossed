import os, os.path
os.chdir("/Users/eklavya/Projects/Amsterdam/EnGlossed")

import numpy as np
import cv2
import os, os.path
from glob import glob

BASE_DIR = 'ENZHZS-F1-EBK'
ITEMS = ('ENG', 'ZH-HANZI', 'ZH-PINYIN', 'ZH-IPA', 'ZS-HANZI', 'ZS-PINYIN', 'ZS-IPA')
OUTPUT_FILENAME = 'GLOSSIKA-FLUENCY3-{number:0>4}-{item}.png'
MAX_SENTENCES_PER_PAGE = 2
BORDER = 10

def slicePage(image):
	# area of the little rectangles next to the sentences, relative to image size
	rect_area = 0.00035 * image.shape[0] * image.shape[1]

	# threashold image and detect contours
	imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(imgray, 250, 255, cv2.THRESH_BINARY)
	_,contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# find the little rectangles next to the sentences
	squares = []
	for cnt in contours:
		cnt_len = cv2.arcLength(cnt, True)
		cnt = cv2.approxPolyDP(cnt, 0.1*cnt_len, True)
		if len(cnt) == 4 and cv2.contourArea(cnt) > rect_area:
			cnt = cnt.reshape(-1, 2)
			squares.append(cnt)

	# check if number of squares fits pattern
	if (len(squares) == 0) or (len(squares) % len(ITEMS) != 0) or (len(squares) / len(ITEMS) > MAX_SENTENCES_PER_PAGE):
		return None

	# items start below the highest point of each square
	cut_y = sorted([min([p[1] for p in sq]) for sq in squares])

	# use right-most point to start looking for the vertical black line
	max_x = max([max([p[0] for p in sq]) for sq in squares]) + 1
	# from the first rectangle, walk right until hitting the line
	line = thresh[cut_y[0], max_x:]
	line_start = np.where( line < 255 )[0][0]
	cut_x = max_x + line_start + np.where( line[line_start:] == 255 )[0][0]
		
	# adjust y-values if they intersect with any of the text
	for i in range(len(cut_y)):
		counter = 10		
		# there should not be any non-white pixels on the vertical line
		while len(np.where( thresh[cut_y[i], cut_x:] < 255 )[0]) > 0 and counter > 0:
			counter -= 1
			cut_y[i] -= 2

	slices = []
	for i in range(len(cut_y)):
		top = cut_y[i]
		bottom = cut_y[i+1] if i+1 < len(cut_y) else image.shape[0]
		left = cut_x; right = image.shape[1]
		area_thresh = thresh[top:bottom, left:right]

		# cut off whitspace
		yBlackRange = np.where(np.min(area_thresh, axis=1) < 255)[0]
		bottom = top + yBlackRange[-1]
		top = top + yBlackRange[0]
		xBlackRange = np.where(np.min(area_thresh, axis=0) < 255)[0]
		right = left + xBlackRange[-1]
		left = left + xBlackRange[0]		
		
		slices.append(image[top:bottom, left:right])

	return slices

# create folders
for item in ITEMS:
	try:
		os.makedirs(os.path.join(BASE_DIR, item))
	except OSError:
		pass

# list image files
files = sorted(glob(os.path.join(BASE_DIR,'*.png')))
if len(files) == 0:
	print('No files found.')
	quit()

sentenceNo = 0
for filename in files:
	print('[+] {}'.format(filename))
	image = cv2.imread(filename)
	slices = slicePage(image)
	if slices is None:
		print('Not a sentence page.')
		continue

	for i, item in enumerate(slices):
		# add border
		shape = list(item.shape)
		shape[0] += 2*BORDER; shape[1] += 2*BORDER
		image = np.zeros(shape, np.uint8) + 255
		image[BORDER:shape[0]-BORDER, BORDER:shape[1]-BORDER] = item

		if i % len(ITEMS) == 0:
			sentenceNo += 1

		outfilename = OUTPUT_FILENAME.format(number=sentenceNo, item=ITEMS[i % len(ITEMS)])
		cv2.imwrite(os.path.join(BASE_DIR, ITEMS[i % len(ITEMS)], outfilename), image)

print('Sliced {} sentences.'.format(sentenceNo))
