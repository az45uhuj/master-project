import os
import sys
import glob

import dlib
import multiprocessing
import cv2


faces_folder = "E:\\3d_morphable_model_199/dlib_dataset/xml/"


def train(race):
    options = dlib.shape_predictor_training_options()
    options.oversampling_amount = 5
    options.nu = 0.1
    options.tree_depth = 4
    options.be_verbose = True
    options.cascade_depth = 15
    options.num_threads = multiprocessing.cpu_count()

    training_xml_path = os.path.join(faces_folder, race + '_train' + ".xml")
    dlib.train_shape_predictor(training_xml_path, faces_folder + race + "_predictor.dat", options)

    print("\nTraining accuracy: {}".format(
        dlib.test_shape_predictor(training_xml_path, faces_folder + race + "_predictor.dat")))

def test(predictor, test_race):
    # The real test is to see how well it does on data it wasn't trained on.  We
    # trained it on a very small dataset so the accuracy is not extremely high, but
    # it's still doing quite good.  Moreover, if you train it on one of the large
    # face landmarking datasets you will obtain state-of-the-art results, as shown
    # in the Kazemi paper.
    #face_folder = 'E:\\facedataset2\dlib_dataset\xml\\'
    face_folder = 'E:\\facedataset2\dlib_dataset\dlibmodel\\'
    testing_xml_path = os.path.join(face_folder, test_race + '_test.xml')
    print("Testing accuracy: {}".format(
        dlib.test_shape_predictor(testing_xml_path, predictor + "_predictor.dat")))

def display():
    # Now let's use it as you would in a normal application.  First we will load it
    # from disk. We also need to load a face detector to provide the initial
    # estimate of the facial location.
    number = 12
    face_folder = 'E:/3d_morphable_model_199/black/img/'+ str(number) + '/'
    predictor = dlib.shape_predictor("E:/3d_morphable_model_199/dlib_dataset/xml/asian_predictor.dat")
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    # Now let's run the detector and shape_predictor over the images in the faces
    # folder and display the results.
    print("Showing detections and predictions on the images in the faces folder...")
    win = dlib.image_window()
    i = 0
    for f in glob.glob(os.path.join(face_folder, "*.png")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        print('before detecting')
        dets = detector(img, 1)
        #print(dets)
        print('after detecting')
        print("Number of faces detected: {}".format(len(dets)))
        if dets:
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
                # Get the landmarks/parts for the face in box d.
                shape = predictor(img, d.rect)
                print(shape.part(0).x)
                p0 = (shape.part(0).x, shape.part(0).y)
                p1 = (shape.part(1).x, shape.part(1).y)
                p2 = (shape.part(2).x, shape.part(2).y)
                p3 = (shape.part(3).x, shape.part(3).y)
                p4 = (shape.part(4).x, shape.part(4).y)

                print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                          shape.part(1)))
                # Draw the face landmarks on the screen.
                win.add_overlay(shape)
            image = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (d.rect.left(), d.rect.top()),  (d.rect.right(), d.rect.bottom()), (255,0,0), 1)
            image = cv2.circle(image, p0, 0, (0, 255, 0), thickness=5)
            image = cv2.circle(image, p1, 0, (0, 255, 0), thickness=5)
            image = cv2.circle(image, p2, 0, (0, 255, 0), thickness=5)
            image = cv2.circle(image, p3, 0, (0, 255, 0), thickness=5)
            image = cv2.circle(image, p4, 0, (0, 255, 0), thickness=5)
            cv2.imshow('1', image)
            cv2.imwrite('E:\\3d_morphable_model_199\dlib_dataset\\asian\\black_' + str(number) + '.png', image)
            cv2.waitKey(0)
            win.add_overlay(dets[0].rect)
            #win.set_image()
        dlib.hit_enter_to_continue()
        i = i + 1

if __name__ == "__main__":
    #train('balanced')
    test('E://3d_morphable_model_199/dlib_dataset/xml/white', 'white')
    #display()