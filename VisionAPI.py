from google.cloud import vision
import io
import os, os.path
import time

DIR = os.path.dirname(os.path.realpath(__file__))

QueueDIR = DIR + "/Queue"
DoneDIR = DIR + "/Done"

vision_client = vision.Client(project='defect-tracking-system')

while(1):
    time.sleep(10)
    for name in os.listdir(QueueDIR):
        print name + " is uploading..."

        filePath = QueueDIR + "/" + name
        with io.open(filePath, 'rb') as image_file:
            content = image_file.read()
        image = vision_client.image(content=content)

        labels = image.detect_labels(limit=10)

        print "Finishing..."
        print "Labels:"
        for label in labels:
            print label.description
            print label.score

        newFilePath = DoneDIR + "/" + name

        os.rename(filePath, newFilePath)
        time.sleep(10)

