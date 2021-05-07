
from digit_images import Digit
import numpy as np
import random
import time

digits = False
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
DATA_SET_SIZE = 0   # initialize the data size

# -------------- add the function later on, limit how many train case we use --------------
# percentage of the info we use
train_info_percentage = 100

# if we are doing training, we set load_digits(0), mode_train == True
# if we are doing validate , we set load_digits(1), mode_validate == True
# if we are doing testing, we set load_digits(2), mode_test == True
mode_train = False
mode_validate = False
mode_test = True

start_time = time.time()

weight = np.zeros((10,29), dtype=float)
for i in range(len(weight)):
    for j in range(len(weight[i])):
        weight[i][j] = random.uniform(-1,1)

previous = np.zeros((10,29), dtype=float)


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
    digit_size = (DIGIT_DATUM_HEIGHT, DIGIT_DATUM_WIDTH)


    # this is the list we actually want, the weights


    # Create a list to hold the predicted label and the actual labels
    predicted_labels = []
    actual_label = []

########################################################################################################################



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

    # Only active when Training
    # it returns a list of length 28
    def count_continuous(array):
        
        count_list = []
        for i in range(28):
            count = 0
            for j in range(27):
                if array[i][j] != 0 and array[i][j+1] != 0:
                    count +=1
                if array[i][j] != 0 and array[i][j+1] == 0:
                    break
            if count >= 5 and count < 8:
                count_list.append(1)
            elif count >=8:
                count_list.append(2)
            elif count < 5:
                count_list.append(0)
        return count_list

    # Only active when Training
    # Here we get the weights based on the model we have
    def calculate_weight (label,array):

        f_value = 0         # f = w0 + w1*common_area1 + w2*common_area2...
        f_value_list = []   # contains f_value for digits from 0 to 9, length of 10

        # Length of 28 count_list[0] = 1 if line 0 has more than 6 coninues cells
        count_list = count_continuous(array)

        # f = w0 + w1*common_area1 + w2*common_area2...

        for digit_label in range(10):
            # calculate f_value for all 10 digits from 0 to 9
            f_value = weight[digit_label][0]       
            for i in range(1,29):
                f_value += (count_list[i-1]* weight[digit_label][i])

            # add the f_value in the list        
            f_value_list.append(f_value)

        predicted_label = f_value_list.index(max(f_value_list))

        # if f_value >= 0 means our weights are good
        # if f_value < 0 means we need to imporve it
        # improvement is given in the project slides as below

        if predicted_label == label: # we get it right
            if f_value_list[predicted_label] < 0:       # right but f < 0, we need to rise the weights 
                weight[label][0] = weight[label][0] + 1
                for i in range(1,29):
                    weight[label][i] = weight[label][i] + count_list[i-1]  

        elif predicted_label != label: # we get it wrong
            if f_value_list[predicted_label] >= 0:       # wrong but f > 0, we need to rise the weights 
                weight[label][0] = weight[label][0] + 1     # we rise the right label
                for i in range(1,29):
                    weight[label][i] = weight[label][i] + count_list[i-1]     

                weight[predicted_label][0] = weight[predicted_label][0] - 1     # we drag the weights for the wrong label down
                for i in range(1,29):
                    weight[predicted_label][i] = weight[predicted_label][i] - count_list[i-1]     


    # Only active when Testing
    def perform_perceptron (label,array):

        f_value = 0         # f = w0 + w1*common_area1 + w2*common_area2...
        f_value_list = []   # contains f_value for digits from 0 to 9, length of 10

        # this is a 10 * 28 2D : the weight model
        model_of_weights = np.loadtxt('Perceptron_Con/weight_model.txt',dtype = 'float')
        count_list = count_continuous(array)

        for digit_label in range(10):
            # calculate f_value for all 10 digits from 0 to 9
            f_value = model_of_weights[digit_label][0]       
            for i in range(1,29):
                f_value += (count_list[i-1]* model_of_weights[digit_label][i])

            # add the f_value in the list        
            f_value_list.append(f_value)

        predicted_labels.append(f_value_list.index(max(f_value_list)))

########################################################################################################################################################
    # load_digits: load every training image and label into a list of tuples (label, array)

    ############################  TRAINING PART  ############################
    if mode_train:

        DATA_SET_SIZE = round(5000 * (.01 * train_info_percentage))

        training_list = load_digits()

        done = False
        counttt = 0
        while(done == False):
            counttt += 1
            print("loop time is:")
            print(counttt)


            for i in range(10):
                for j in range(29):
                    previous[i][j] = weight[i][j]


            # we now have the model, we need to add weights on
            for i in training_list:
                array = i.get_array()   #current image
                label = i.get_label()   #the label of current image
                calculate_weight(label,array)
            
            done = True

            for r in range(10):
                for c in range(29):
                    if previous[r][c] != weight[r][c] and counttt <50:
                        done = False



        print("--------- PERCEPTRON TRAINING COMPLETE --------- ")
        print("The percentage of training images we used: " + str(train_info_percentage) + "%")
        print("Total training images used: "  + str(DATA_SET_SIZE))
        print("TOTAL TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))

        # output the model matrix
        np.savetxt('Perceptron_Con/weight_model.txt',weight,fmt='%0.5f')



    ############################  VALIDATION PART  ############################
    if mode_validate:

        validation_list = load_digits()
        # Call each individual training image individually. Pass to method to populate density array
        for i in validation_list:
            array = i.get_array()   #current image
            label = i.get_label()   #the label of current image
            actual_label.append(label)
            perform_perceptron(label,array)

    

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
        print("--------- PERCEPTRON TEST COMPLETE --------- ")
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness*100))+"%")

    ############################ TESTING PART  ############################

    if mode_test:

        validation_list = load_digits()
        # Call each individual training image individually. Pass to method to populate density array
        for i in validation_list:
            array = i.get_array()   #current image
            label = i.get_label()   #the label of current image
            actual_label.append(label)
            perform_perceptron(label,array)

    

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
        print("--------- PERCEPTRON TEST COMPLETE --------- ")
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness*100))+"%")


































########################################################################################################################
#######################################   PERCEPTRON FACES #############################################################



else:



    new_weight = np.zeros((2, 3), dtype=float)
    for i in range(len(new_weight)):
        for j in range(len(new_weight[i])):
            new_weight[i][j] = random.uniform(-1, 1)

    new_previous = np.zeros((2, 3), dtype=float)

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




    def get_max_length(array):

        max_length = 0


        for m in range(0, FACE_DATUM_HEIGHT):
            row_length = 0
            for n in range(0, FACE_DATUM_WIDTH):
                if array[m,n] == 1:
                    row_length += 1
                    if row_length > max_length:
                        max_length = row_length
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



# Only active when Training
    # Here we get the weights based on the model we have
    def calculate_weight (label,array):

        f_value = 0         # f = w0 + w1*common_area1 + w2*common_area2...
        f_value_list = []   # contains not faces, faces. length of 2

        # Length of 28 count_list[0] = 1 if line 0 has more than 6 coninues cells
        max_row = get_max_length(array)
        max_colum = get_max_column(array)
        #count_list = count_continuous(array)

        # f = w0 + w1*common_area1 + w2*common_area2...

        for digit_label in range(2):
            # calculate f_value for all 10 digits from 0 to 9
            f_value = weight[digit_label][0]
            for i in range(1,3):
                f_value += max_row * weight[digit_label][i]

            # add the f_value in the list
            f_value_list.append(f_value)

        predicted_label = f_value_list.index(max(f_value_list))

        # if f_value >= 0 means our weights are good
        # if f_value < 0 means we need to imporve it
        # improvement is given in the project slides as below

        if predicted_label == label: # we get it right
            if f_value_list[predicted_label] < 0:       # right but f < 0, we need to rise the weights
                weight[label][0] = weight[label][0] + 1
                for i in range(1,3):
                    weight[label][i] = weight[label][i] + max_row

        elif predicted_label != label: # we get it wrong
            if f_value_list[predicted_label] >= 0:       # wrong but f > 0, we need to rise the weights
                weight[label][0] = weight[label][0] + 1     # we rise the right label
                for i in range(1,3):
                    weight[label][i] = weight[label][i] + max_row

                weight[predicted_label][0] = weight[predicted_label][0] - 1     # we drag the weights for the wrong label down
                for i in range(1,3):
                    weight[predicted_label][i] = weight[predicted_label][i] - max_row




    # Only active when Testing
    def perform_perceptron (label,array):

        f_value = 0         # f = w0 + w1*common_area1 + w2*common_area2...
        f_value_list = []   # contains f_value for digits from 0 to 9, length of 10

        # this is a 10 * 28 2D : the weight model
        model_of_weights = np.loadtxt('Perceptron_Con/weight_model_faces.txt',dtype = 'float')
        # count_list = count_continuous(array)
        max_row = get_max_length(array)
        max_columns = get_max_column(array)

        for digit_label in range(2):
            # calculate f_value for all 10 digits from 0 to 9
            f_value = model_of_weights[digit_label][0]
            for i in range(1,3):
                f_value += max_row * model_of_weights[digit_label][i]

            # add the f_value in the list
            f_value_list.append(f_value)

        predicted_labels.append(f_value_list.index(max(f_value_list)))






    if mode_train:

        DATA_SET_SIZE = round(451 * (.01 * train_info_percentage))

        training_list = load_faces()



        done = False
        counttt = 0
        while (done == False):
            counttt += 1
            print("loop time is:")
            print(counttt)

            for i in range(2):
                for j in range(3):
                    previous[i][j] = weight[i][j]

            # we now have the model, we need to add weights on
            for i in training_list:
                array = i.get_array()  # current image
                label = i.get_label()  # the label of current image
                calculate_weight(label, array)

            done = True

            for r in range(2):
                for c in range(3):
                    if previous[r][c] != weight[r][c] and counttt < 50:
                        done = False

        print("--------- PERCEPTRON TRAINING COMPLETE --------- ")
        print("The percentage of training images we used: " + str(train_info_percentage) + "%")
        print("Total training images used: " + str(DATA_SET_SIZE))
        print("TOTAL TRAINING TIME: --- %s seconds ---" % (time.time() - start_time))

        # output the model matrix
        np.savetxt('Perceptron_Con/weight_model_faces.txt', weight, fmt='%0.5f')

    ############################  VALIDATION PART  ############################
    if mode_validate:

        validation_list = load_faces()
        # Call each individual training image individually. Pass to method to populate density array
        for i in validation_list:
            array = i.get_array()  # current image
            label = i.get_label()  # the label of current image
            actual_label.append(label)
            perform_perceptron(label, array)

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
        print("--------- PERCEPTRON VALIDATION TEST COMPLETE --------- ")
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness * 100)) + "%")

    ############################ TESTING PART  ############################

    if mode_test:

        validation_list = load_faces()
        # Call each individual training image individually. Pass to method to populate density array
        for i in validation_list:
            array = i.get_array()  # current image
            label = i.get_label()  # the label of current image
            actual_label.append(label)
            perform_perceptron(label, array)

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
        print("--------- PERCEPTRON TESTING COMPLETE --------- ")
        print("\tCORRECT GUESSES: " + str(correct))
        print("\tINCORRECT GUESSES: " + str(incorrect))
        print("\tCorrectness: " + str(int(Correctness * 100)) + "%")



