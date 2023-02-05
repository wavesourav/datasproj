#Imports
import PIL
from my_package.model import ImageCaptioningModel
from my_package.data import Dataset, Download
from my_package.data.transforms import FlipImage, RescaleImage, BlurImage, CropImage, RotateImage
import numpy as np
from PIL import Image


def experiment(annotation_file, captioner, transforms, outputs):
    '''
        Function to perform the desired experiments
        Arguments:
        annotation_file: Path to annotation file
        captioner: The image captioner
        transforms: List of transformation classes
        outputs: Path of the output folder to store the images
    '''

    #Create the instances of the dataset, download
    data = Dataset(annotation_file, transforms)
    down = Download()

    #Print image names and their captions from annotation file using dataset object
    # for item in data.jsonList:
    #     print(item["file_name"]+"\n")
    #     for i in item["captions"]:
    #         print(i["caption"]+"\n")


    #Download images to ./data/imgs/ folder using download object
    # for item in data.jsonList:
    #     fn="./data/imgs/"+item["file_name"]
    #     url=item["url"]
    #     down(fn,url)


    #Transform the required image (roll number mod 10) and save it seperately
    img=data.__getann__(9)
    data.__transformitem__(img,outputs)




    #Get the predictions from the captioner for the above saved transformed image

    print(captioner(outputs,3))






def main():
    captioner = ImageCaptioningModel()

    experiment('./data/annotations.jsonl', captioner, [],"./output/9_A.jpg") # Sample arguments to call experiment()
    print("First Analysis is Completed")
    experiment('./data/annotations.jsonl', captioner, [FlipImage('horizontal')], "Output/9_B.jpg")
    print("Second Analysis is Completed")
    experiment('./data/annotations.jsonl', captioner, [BlurImage(50)], "output/9_C.jpg")
    print("Third Analysis is Completed")
    experiment('./data/annotations.jsonl', captioner, [RescaleImage(2)], "output/9_D.jpg")
    print("Fourth Analysis is Completed")
    experiment('./data/annotations.jsonl', captioner, [RescaleImage(0.5)], "output/9_E.jpg")
    print("Fifth Analysis is Completed")
    experiment('./data/annotations.jsonl', captioner, [RotateImage(-90)], "output/9_F.jpg")
    print("Sixth Analysis is Completed")
    experiment('./data/annotations.jsonl', captioner, [RotateImage(45)], "output/9_G.jpg")
    print("Seventh Analysis is Completed")


if __name__ == '__main__':
    main()