
from digit_images import Digit
import numpy as np
import time

digits = True
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
DATA_SET_SIZE = 0   # initialize the data size

# -------------- add the function later on, limit how many train case we use --------------
# percentage of the info we use
train_info_percentage = 50

# if we are doing training, we set load_digits(0), mode_train == True
# if we are doing validate , we set load_digits(1), mode_validate == True
# if we are doing testing, we set load_digits(2), mode_test == True
mode_train = False
mode_validate = False
mode_test = True

start_time = time.time()

# =================================== DIGIT PART =================================== 
# get the data size of digit
if digits:
    if mode_train:
        l = open("digitdata/traininglabels", "r")
    if mode_validate:
        l = open("digitdata/validationlabels", "r")
    if mode_test:
        l = open("digitdata/testlabels", "r")

    # total_label_counter: total number of labels
    total_label_counter = 0

    for line in l:
        line = line.strip("\n")
        total_label_counter += 1

    DATA_SET_SIZE = total_label_counter 
    #print("this is how many images we have:" + str(DATA_SET_SIZE))

    l.close()


    # digit_size - the size of the image 2d array
    digit_size = (DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)


    # Create a density array for each classifier
    # These are the models we have for digits from 0 - 9 (2D-array)
    density_0 = np.zeros((digit_size), dtype=int)
    density_1 = np.zeros((digit_size), dtype=int)
    density_2 = np.zeros((digit_size), dtype=int)
    density_3 = np.zeros((digit_size), dtype=int)
    density_4 = np.zeros((digit_size), dtype=int)
    density_5 = np.zeros((digit_size), dtype=int)
    density_6 = np.zeros((digit_size), dtype=int)
    density_7 = np.zeros((digit_size), dtype=int)
    density_8 = np.zeros((digit_size), dtype=int)
    density_9 = np.zeros((digit_size), dtype=int)
    # Put model arrays into a list
    # density_list -> the whole model we have to recognize digits
    density_list = [density_0,density_1,density_2,density_3,density_4,density_5,density_6,density_7,density_8,density_9]

    # Create a list to hold the predicted label and the actual labels
    predicted_labels = []
    actual_label = []


    # get all the images
    def load_digits():
        # A list that holds a tuple for each image.    (label, 2D numpy array)
        t_list = []
        if mode_train:
            d = open("digitdata/trainingimages", "r")
            l = open("digitdata/traininglabels", "r")
        if mode_validate:
            d = open("digitdata/validationimages", "r")
            l = open("digitdata/validationlabels", "r")
        elif mode_test:
            d = open("digitdata/testimages", "r")
            l = open("digitdata/testlabels", "r")

        for i in range(DATA_SET_SIZE):
            # get the label so we know what we are looking at
            label = l.readline()
            label = label.strip()
            label = int(label)

            # temp_list - where we store out current image
            temp_list = []
            for j in range(DIGIT_DATUM_HEIGHT):
                temp_row = []
                for k in range(DIGIT_DATUM_WIDTH):
                    bit = d.read(1)
                    if bit == "+":
                        temp_row.append(1)
                    elif bit == "#":
                        temp_row.append(2)
                    elif bit == "\n":
                        bit = d.read(1)
                        if bit == "+":
                            temp_row.append(1)
                        elif bit == "#":
                            temp_row.append(2)
                        elif bit.isspace():
                            temp_row.append(0)
                    elif bit.isspace():
                        temp_row.append(0)

                temp_list.append(temp_row)

            np_array = np.array(temp_list)
            image_data = Digit(label, np_array)
            t_list.append(image_data)

        d.close()
        l.close()
        return t_list


    def calculate_pc():
        # Make a new list to hold all of the labels
        label_list = []

        # Make a list to hold the p(c) value
        # prior_list [P(C=1), P(C=2), ..., P(C=9)]
        prior_list = []

        # Open the  label file
        if mode_train:
            l = open("digitdata/traininglabels", "r")
        if mode_validate:
            l = open("digitdata/validationlabels", "r")
        if mode_test:
            l = open("digitdata/testlabels", "r")

        # Extract all the labels up until the Test_Set_Size
        # total_label_counter - total number of labels
        total_label_counter = 0

        for i in l:
            label_list.append(i)
            total_label_counter += 1

        # Loop through every classifier
        for j in range(10):
            counter = 0
            for k in label_list:
                if int(k) == j:
                    counter += 1
            prior_list.append(counter/total_label_counter)

        l.close()
        return prior_list


    # add more layer on the model, make the common area darker and darker
    def populate_density(label, array):

        # get the density_label, update it
        # if the label is 1, we will get density_1 which is the current temp model of 1
        density_array = density_list[label]

        for i in range(0,DIGIT_DATUM_HEIGHT ):
            for j in range(0, DIGIT_DATUM_WIDTH):
                if array[i, j] == 1:
                    density_array[i,j] = density_array[i,j] + 1
                elif array[i, j] == 2:
                    density_array[i,j] = density_array[i,j] + 5

        density_list[label] = density_array


    # test_array is the array we are looking at now
    # we check the array among the 9 models
    # then pick the highest possibility
    def perform_bayes(test_array):
        # List to hold the probability percentage for each classifier
        prob_list  = []

        # Loop through all 10 classifiers
        for i in range(10):
            #read the models
            filename = 'BayesModelDigits/model_%d.txt' % i
            density_array = np.loadtxt(filename,dtype = 'int')

            temp_probability = 1

            common_cells_1 = 0

            for m in range(DIGIT_DATUM_HEIGHT):
                for n in range(DIGIT_DATUM_WIDTH):
                    if density_array[m,n] != 0 and test_array[m,n] != 0:
                        common_cells_1 += 1
                    elif density_array[m,n] == 0 and test_array[m,n] == 0:
                        common_cells_1 += 1
                temp_probability *= ((common_cells_1 / 28) + 0.000001)

            prob_list.append(temp_probability * pc_list[i])

        predicted_labels.append(prob_list.index(max(prob_list)))


    # load_digits: load every training image and label into a list of tuples (label, array)

    ############################  TRAINING PART  ############################
    if mode_train:

        DATA_SET_SIZE = round(5000 * (.01 * train_info_percentage))

        training_list = load_digits()
        # Call each individual training image individually. Pass to method to populate density array
        for i in training_list:
            array = i.get_array()   #current image
            label = i.get_label()   #the label of current image
            populate_density(label,array)


        for density_x in density_list:
            for i in range(len(density_x)):
                for j in range(len(density_x[i])):
                    if density_x[i][j] < 650:
                        density_x[i][j]  = 0

        print("--------- NAIVE BAYES TRAINING COMPLETE --------- ")
        print("The percentage of training images we used: " + str(train_info_percentage) + "%")
        print("Total training images used: "  + str(DATA_SET_SIZE))
        print("TOTAL TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))

        # output the model matrix
        np.savetxt('BayesModelDigits/model_0.txt',density_0,fmt='%d')
        np.savetxt('BayesModelDigits/model_1.txt',density_1,fmt='%d')
        np.savetxt('BayesModelDigits/model_2.txt',density_2,fmt='%d')
        np.savetxt('BayesModelDigits/model_3.txt',density_3,fmt='%d')
        np.savetxt('BayesModelDigits/model_4.txt',density_4,fmt='%d')
        np.savetxt('BayesModelDigits/model_5.txt',density_5,fmt='%d')
        np.savetxt('BayesModelDigits/model_6.txt',density_6,fmt='%d')
        np.savetxt('BayesModelDigits/model_7.txt',density_7,fmt='%d')
        np.savetxt('BayesModelDigits/model_8.txt',density_8,fmt='%d')
        np.savetxt('BayesModelDigits/model_9.txt',density_9,fmt='%d')


    ############################  VALIDATION PART  ############################
    if mode_validate:

        # Calculate the P(C) value for each C. Store in a 10 index array
        pc_list = calculate_pc()
        validate_list = load_digits()
        for i in validate_list:
            actual_label.append(i.get_label())

        # Perform Bayes algorithm to guess the value for every test image
        for j in validate_list:
            perform_bayes(j.get_array())



        # Compare predicted digit vs. actual label
        correct = 0
        incorrect = 0
        for k in range(len(actual_label)):
            if actual_label[k] == predicted_labels[k]:
                correct += 1
            else:
                incorrect += 1

        Correctness = correct/DATA_SET_SIZE


        # OUTPUT
        print("--------- NAIVE BAYES TEST COMPLETE --------- ")
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness*100))+"%")

    ############################ TESTING PART  ############################
    if mode_test:

        # Calculate the P(C) value for each C. Store in a 10 index array
        pc_list = calculate_pc()
        test_list = load_digits()
        for i in test_list:
            actual_label.append(i.get_label())

        # Perform Bayes algorithm to guess the value for every test image
        for j in test_list:
            perform_bayes(j.get_array())

        # Compare predicted digit vs. actual label
        correct = 0
        incorrect = 0
        for k in range(len(actual_label)):
            if actual_label[k] == predicted_labels[k]:
                correct += 1
            else:
                incorrect += 1

        Correctness = correct/DATA_SET_SIZE


        # OUTPUT
        print("--------- NAIVE BAYES TEST COMPLETE --------- ")
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness*100))+"%")








# =================================== FACE PART ===================================
# =================================================================================

else:
    max_lengths_0 = np.array([])
    max_lengths_1 = np.array([])
    max_columns_0 = np.array([])
    max_columns_1 = np.array([])

    if mode_train:
        l = open("facedata/facedatatrainlabels", "r")
    if mode_validate:
        l = open("facedata/facedatavalidationlabels", "r")
    if mode_test:
        l = open("facedata/facedatatestlabels", "r")

        # total_label_counter: total number of labels
    total_label_counter = 0

    for line in l:
        line = line.strip("\n")
        total_label_counter += 1

    DATA_SET_SIZE = total_label_counter

    l.close()

    # digit_size - the size of the image 2d array
    face_size = (FACE_DATUM_HEIGHT, FACE_DATUM_WIDTH)

    # Put model arrays into a list
    # density_list -> the whole model we have to recognize digits

    # Create a list to hold the predicted label and the actual labels
    predicted_labels = []
    actual_label = []

# get all the images
    def load_faces():
        # A list that holds a tuple for each image.    (label, 2D numpy array)
        t_list = []
        if mode_train:
            d = open("facedata/facedatatrain", "r")
            l = open("facedata/facedatatrainlabels", "r")
        if mode_validate:
            d = open("facedata/facedatavalidation", "r")
            l = open("facedata/facedatavalidationlabels", "r")
        elif mode_test:
            d = open("facedata/facedatatest", "r")
            l = open("facedata/facedatatestlabels", "r")

        for i in range(DATA_SET_SIZE):
            # get the label so we know what we are looking at
            label = l.readline()
            label = label.strip()
            label = int(label)

            # temp_list - where we store out current image
            temp_list = []
            for j in range(FACE_DATUM_HEIGHT):
                temp_row = []
                for k in range(FACE_DATUM_WIDTH):
                    bit = d.read(1)
                    if bit == "#":
                        temp_row.append(1)
                    elif bit == "\n":
                        bit = d.read(1)
                        if bit == "#":
                            temp_row.append(1)
                        elif bit.isspace():
                            temp_row.append(0)
                    elif bit.isspace():
                        temp_row.append(0)

                temp_list.append(temp_row)

            np_array = np.array(temp_list)
            image_data = Digit(label, np_array)
            t_list.append(image_data)

        d.close()
        l.close()
        return t_list



    def calculate_pc():
        # Make a new list to hold all of the labels
        label_list = []

        # Make a list to hold the p(c) value
        # prior_list [P(C=1), P(C=2), ..., P(C=9)]
        prior_list = []

        # Open the  label file
        if mode_train:
            l = open("facedata/facedatatrainlabels", "r")
        if mode_validate:
            l = open("facedata/facedatavalidationlabels", "r")
        if mode_test:
            l = open("facedata/facedatatestlabels", "r")

        # Extract all the labels up until the Test_Set_Size
        # total_label_counter - total number of labels
        total_label_counter = 0

        for i in l:
            label_list.append(i)
            total_label_counter += 1

        # Loop through every classifier
        for j in range(2):
            counter = 0
            for k in label_list:
                if int(k) == j:
                    counter += 1
            prior_list.append(counter/total_label_counter)

        l.close()
        return prior_list





    def get_max_length(array):

        max_length = 0

        for m in range(0, FACE_DATUM_HEIGHT):
            row_length = 0
            for n in range(0, FACE_DATUM_WIDTH):
                if array[m,n] == 1:
                    row_length += 1
                    if row_length > max_length:
                        max_length = row_length
                        count = 0
                else:
                    row_length = 0

        return max_length




    def get_max_column(array):

        max_length = 0

        for m in range(0, FACE_DATUM_WIDTH):
            column_max = 0
            for n in range(0, FACE_DATUM_HEIGHT):
                if array[n,m] == 1:
                    column_max += 1
                    if column_max > max_length:
                        max_length = column_max
                else:
                    column_max = 0

        return max_length



    def perform_bayes_1(test_array):
        # List to hold the probability percentage for each classifier
        prob_list  = []

        # Get max length of test images
        test_row_max = get_max_length(test_array)
        test_column_max = get_max_column(test_array)

        # Loop through both classifiers
        for i in range(2):
            count_1 = 0
            count_2 = 0

            #read the models
            filename_row = 'BayesModelFaces/model_%d.txt' % i
            filename_column = 'BayesModelFaces/columns_%d.txt' % i

            max_lengths = np.loadtxt(filename_row,dtype = 'int')
            max_columns = np.loadtxt(filename_column,dtype = 'int')


            for m in max_lengths:
                if m == test_row_max:
                    count_1 += 1

            for n in max_columns:
                if n == test_column_max:
                    count_2 += 1


            prob_1 = count_1 / 150
            prob_2 = count_2 / 150
            prob_list.append(prob_1 * prob_2 * pc_list[i])



        predicted_labels.append(prob_list.index(max(prob_list)))






# load_digits: load every training image and label into a list of tuples (label, array)
# ______________________________________________________________________________________________________________________
    ############################  TRAINING PART  ############################
    if mode_train:
        training_list = load_faces()
        # Call each individual training image individually. Pass to method to populate density array

        counter = 0
        for m in training_list:
            if counter < round(451 * (train_info_percentage * .01)):
                counter += 1
                array = m.get_array()
                label = m.get_label()
                max_length = get_max_length(array)
                max_column = get_max_column(array)

                if label == 0:
                    max_lengths_0 = np.append(max_lengths_0, max_length)
                    max_columns_0 = np.append(max_columns_0, max_column)
                if label == 1:
                    max_lengths_1 = np.append(max_lengths_1, [max_length])
                    max_columns_1 = np.append(max_columns_1, [max_column])

        np.savetxt('BayesModelFaces/model_0.txt', max_lengths_0, fmt='%d')
        np.savetxt('BayesModelFaces/model_1.txt', max_lengths_1, fmt='%d')
        np.savetxt('BayesModelFaces/columns_0.txt', max_lengths_1, fmt='%d')
        np.savetxt('BayesModelFaces/columns_1.txt', max_lengths_1, fmt='%d')



        print("--------- NAIVE BAYES FACE TRAINING COMPLETE --------- ")
        print("The percentage of training images we use: " + str(train_info_percentage) + "%")
        print("Total IMAGES USED IN TRAINING: " + str(round(451 * (train_info_percentage * .01))))
        print("TOTAL TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))










    ############################  VALIDATION PART  ############################
    if mode_validate:

        # Calculate the P(C) value for each C. Store in a 10 index array
        pc_list = calculate_pc()
        validate_list = load_faces()
        for i in validate_list:
            actual_label.append(i.get_label())

        # Perform Bayes algorithm to guess the value for every test image
        for j in validate_list:
            perform_bayes_1(j.get_array())


        # Compare predicted digit vs. actual label
        correct = 0
        incorrect = 0
        for k in range(len(actual_label)):

            if actual_label[k] == predicted_labels[k]:
                correct += 1
            else:
                incorrect += 1

        Correctness = correct / DATA_SET_SIZE


        # OUTPUT
        print("--------- NAIVE BAYES FACE VALIDATION TEST COMPLETE --------- ")
        print("TOTAL IMAGES VALIDATED: " + str(DATA_SET_SIZE))
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness * 100)) + "%")

    ############################ TESTING PART  ############################

    if mode_test:

        # Calculate the P(C) value for each C. Store in a 10 index array
        pc_list = calculate_pc()
        validate_list = load_faces()
        for i in validate_list:
            actual_label.append(i.get_label())

        # Perform Bayes algorithm to guess the value for every test image
        for j in validate_list:
            perform_bayes_1(j.get_array())

        # Compare predicted digit vs. actual label
        correct = 0
        incorrect = 0
        for k in range(len(actual_label)):

            if actual_label[k] == predicted_labels[k]:
                correct += 1
            else:
                incorrect += 1

        Correctness = correct / DATA_SET_SIZE


        # OUTPUT
        print("--------- NAIVE BAYES FACE TESTING COMPLETE --------- ")
        print("TOTAL IMAGES TESTED:" + str(DATA_SET_SIZE))
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness * 100)) + "%")
