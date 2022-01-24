import json

import cv2
import dlib
from os import listdir
from os.path import isfile, join

from color2gray.color2gray import *
from contrast_enhancement.contrast_enhancement import *

datasets_location = [
    "../datasets/300W/01_Indoor/",
    "../datasets/300W/02_Outdoor/"
]

models = {
    "default_ert": "../models/shape_predictor_68_face_landmarks.dat",
    "improve_ert": "../models/shape_predictor_68_face_landmarks_GTX.dat",
    "default_lfb": "../models/default_opencv_lfbmodel.yaml",
    "improve_lfb": "../models/LBF555_GTX.yaml"
}


detection_errors = {
    "indoor": {},
    "outdoor": {}
}


def read_reference(path, name):
    f = open(path + name + ".pts", "r")

    # Version line
    f.readline()

    # Number of points
    no_points = int(f.readline().split(":")[1])
    landmarks = np.zeros((no_points, 2), dtype="int")

    # {
    f.readline()

    # Read each point
    for i in range(0, no_points):
        line = f.readline().split(" ")
        x = int(float(line[0]))
        y = int(float(line[1]))
        landmarks[i] = (x, y)

    f.close()

    return landmarks


def write_error(value, dataset_type,  ce_tech=0, gray_tech=0, ce_first=False):
    pipeline_identifier = str(ce_tech) + str(gray_tech) + str(int(ce_first))
    if pipeline_identifier not in detection_errors[dataset_type]:
        detection_errors[dataset_type][pipeline_identifier] = [value]
    else:
        detection_errors[dataset_type][pipeline_identifier].append(value)


def compute_error(landmarks, landmarks_reference, dataset_type,  ce_tech=0, gray_tech=0, ce_first=False):
    no_points = landmarks.shape[0]
    no_points_reference = landmarks_reference.shape[0]

    if no_points != no_points_reference:
        print("Different np sizes")
        return

    interocular_distance = 0
    if no_points == 68:
        interocular_distance = np.linalg.norm(landmarks_reference[37, :] - landmarks_reference[46, :])
    else:
        print("Incorrect number of points")
        return

    sum_points = 0
    for i in range(0, no_points):
        sum_points += np.linalg.norm(landmarks[i, :] - landmarks_reference[i, :])

    err = sum_points / (no_points * interocular_distance)
    write_error(err, dataset_type, ce_tech, gray_tech, ce_first)

    return err


MAX_GRAY_TECHNIQUES = 8
MAX_CE_TECHNIQUES = 11


def contrast_enhancement_switch(img, contrast_enhancement_technique):
    # Image enhancement
    # Contrast enhancement techniques
    res = img
    # HE
    if contrast_enhancement_technique == 1:
        res = contrast_enhancement(img, CType.USE_COLOR, cv2.equalizeHist)
    elif contrast_enhancement_technique == 2:
        res = contrast_enhancement(img, CType.USE_LAB_L, cv2.equalizeHist)
    # gamma transform
    elif contrast_enhancement_technique == 3:
        res = contrast_enhancement(img, CType.USE_COLOR, gamma_transform)
    elif contrast_enhancement_technique == 4:
        res = contrast_enhancement(img, CType.USE_LAB_L, gamma_transform)
    # MIN-MAX
    elif contrast_enhancement_technique == 5:
        res = contrast_enhancement(img, CType.USE_COLOR, min_max_contrast_stretching)
    elif contrast_enhancement_technique == 6:
        res = contrast_enhancement(img, CType.USE_LAB_L, min_max_contrast_stretching)
    # CLAHE
    elif contrast_enhancement_technique == 7:
        res = contrast_enhancement(img, CType.USE_COLOR, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply)
    elif contrast_enhancement_technique == 8:
        res = contrast_enhancement(img, CType.USE_LAB_L, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply)
    # WTHE
    elif contrast_enhancement_technique == 9:
        res = contrast_enhancement(img, CType.USE_COLOR, weighted_threshold_histogram_equalization)
    elif contrast_enhancement_technique == 10:
        res = contrast_enhancement(img, CType.USE_LAB_L, weighted_threshold_histogram_equalization)
    return res


def gray_switch(img, color2gray_technique):
    # Grayscale conversion
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if color2gray_technique == 1:
        res = pca_color2gray(img)
    elif color2gray_technique == 2:
        res = gray_average(img)
    elif gray_technique == 3:
        res = gray_desaturate(img)
    elif gray_technique == 4:
        res = gray_decompose(img, 'min')
    elif gray_technique == 5:
        res = gray_decompose(img, 'max')
    elif gray_technique == 6:
        res = gray_custom_shades(img, 125)
    elif gray_technique == 7:
        res = gray_variance(img, (0.33, 0.33, 0.34))
    return res


def landmark_detection(path, name,
                       dataset_type,
                       ce_tech=0, gray_tech=0, ce_first=False,
                       draw_orig=False, draw_ref=False, draw_out=False):
    image = cv2.imread(path + name + ".png")

    # Contrast Enhancement First
    if ce_first:
        image = contrast_enhancement_switch(image, ce_tech)

    # Grayscale conversion
    gray = gray_switch(image, gray_tech)

    # Contrast Enhancement First
    if not ce_first:
        image = contrast_enhancement_switch(image, ce_tech)

    # Face detector
    boxes = detector(gray, 1)
    if len(boxes) == 0:
        write_error(0.5, dataset_type, ce_tech, gray_tech, ce_first)

    image_landmarks = np.empty((0, 2), dtype="int")
    reference_landmarks = np.empty((0, 2), dtype="int")

    # Landmark detection
    for box in boxes:
        landmarks = predictor(gray, box)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
        landmarks = shape_np

        landmarks_reference = read_reference(path, name)
        reference_landmarks = np.concatenate((reference_landmarks, landmarks_reference), axis=0)
        image_landmarks = np.concatenate((landmarks, image_landmarks), axis=0)

        compute_error(landmarks, landmarks_reference, dataset_type, ce_tech, gray_tech, ce_first)

        # break after first detection
        break

    if draw_orig:
        cv2.imshow("Image original", image)

    # Reference
    if draw_ref:
        image_reference = image.copy()
        for _, (x, y) in enumerate(reference_landmarks):
            cv2.circle(image_reference, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("Image reference", image_reference)

    if draw_out:
        image_output = image.copy()
        for _, (x, y) in enumerate(image_landmarks):
            cv2.circle(image_output, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("Image", image_output)

    if draw_orig or draw_ref or draw_out:
        cv2.waitKey(0)


def get_images_names(path):
    return [image.split(".")[0] for image in listdir(path)
            if isfile(join(path, image)) and image.endswith(".png")]


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(models["default_ert"])

    for ce_technique in range(0, MAX_CE_TECHNIQUES):
        for gray_technique in range(0, MAX_GRAY_TECHNIQUES):
            for ce_firstly in [True, False]:

                if not ce_firstly and ce_technique % 2 == 0:
                    continue

                # INDOOR
                image_names = get_images_names(datasets_location[0])
                for idx, image_name in enumerate(image_names):
                    landmark_detection(datasets_location[0], image_name, "indoor",
                                       ce_tech=ce_technique,
                                       gray_tech=gray_technique,
                                       ce_first=ce_firstly)

                # OUTDOOR
                image_names = get_images_names(datasets_location[1])
                for idx, image_name in enumerate(image_names):
                    landmark_detection(datasets_location[1], image_name, "outdoor",
                                       ce_tech=ce_technique,
                                       gray_tech=gray_technique,
                                       ce_first=ce_firstly)

                with open("error.txt", "w") as file:
                    file.write(json.dumps(detection_errors, indent=4))
