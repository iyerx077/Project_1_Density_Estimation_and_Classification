import numpy
import scipy.io
import math
import geneNewData


# Numpyfile= scipy.io.loadmat("matlabfile.mat")

def main():
    myID = '1670'  # change to last 4 digit of your studentID
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train' + myID + '.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train' + myID + '.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset' + '.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset' + '.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')

    # Task 1
    train0set = []
    train1set = []

    # extracting datapoints into 2d arrays as [mean,variance]
    for i in range(len(train0)):  # for both tests since train0 and train1 have the same lenght
        train0set.append([numpy.mean(train0[i]), numpy.std(train0[i])])
        train1set.append([numpy.mean(train1[i]), numpy.std(train1[i])])

    # Task 2
    # using axis=0 since it will calculate the same indexed elements for all of the arrays
    meantrain0 = numpy.mean(train0set, axis=0)
    vartrain0 = numpy.var(train0set, axis=0)
    meantrain1 = numpy.mean(train1set, axis=0)
    vartrain1 = numpy.var(train1set, axis=0)

    mf1train0 = meantrain0[0]
    vf1train0 = vartrain0[0]
    mf2train0 = meantrain0[1]
    vf2train0 = vartrain0[1]
    mf1train1 = meantrain1[0]
    vf1train1 = vartrain1[0]
    mf2train1 = meantrain1[1]
    vf2train1 = vartrain1[1]

    # Task 3
    test0set = []
    test1set = []

    # extracting datapoints into 2d arrays as [mean,standard deviation]
    for j in range(len(test0)):  # for both tests since train0 and train1 have the same lenght
        test0set.append([numpy.mean(test0[j]), numpy.std(test0[j])])

    for t in range(len(test1)):
        test1set.append([numpy.mean(test1[t]), numpy.std(test1[t])])

    samplespredicted1 = 0
    samplespredicted0 = 0

    # compute P(X|y=0) by multiplying the PDF with the input test datapoints for std and mean and probability of each digit
    for f in range(len(test0set)):
        probf1train0 = NB(test0set[f][0], mf1train0, math.sqrt(vf1train0))
        probf2train0 = NB(test0set[f][1], mf2train0, math.sqrt(vf2train0))
        expvaluetrain0 = probf1train0 * probf2train0 * 0.5

        probf1train1 = NB(test0set[f][0], mf1train1, math.sqrt(vf1train1))
        probf2train1 = NB(test0set[f][1], mf2train1, math.sqrt(vf2train1))
        expvaluetrain1 = probf1train1 * probf2train1 * 0.5

        # e

        if expvaluetrain0 > expvaluetrain1:
            samplespredicted0 += 1

    for k in range(len(test1set)):
        probf1train0 = NB(test1set[k][0], mf1train0, math.sqrt(vf1train0))
        probf2train0 = NB(test1set[k][1], mf2train0, math.sqrt(vf2train0))
        expectedprobtrain0 = probf1train0 * probf2train0 * 0.5

        probf1train1 = NB(test1set[k][0], mf1train1, math.sqrt(vf1train1))
        probf2train1 = NB(test1set[k][1], mf2train1, math.sqrt(vf2train1))
        expectedprobtrain1 = probf1train1 * probf2train1 * 0.5

        # e

        if expectedprobtrain0 < expectedprobtrain1:
            samplespredicted1 += 1

    # Task 4
    Accuracy_for_digit0testset = samplespredicted0 / len(test0set)
    Accuracy_for_digit1testset = samplespredicted1 / len(test1set)

    # implement NB calssifiers parameters from task 2; use classifiers to predict unknown labels
    # predict labels for test set?

    print(['1670', mf1train0, vf1train0, mf2train0, vf2train0, mf1train1, vf1train1, mf2train1, vf2train1,
           Accuracy_for_digit0testset, Accuracy_for_digit1testset])
    # ['ASUId', Mean_of_feature1_for_digit0, Variance_of_feature1_for_digit0, Mean_of_feature2_for_digit0, Variance_of_feature2_for_digit0 , Mean_of_feature1_for_digit1, Variance_of_feature1_for_digit1, Mean_of_feature2_for_digit1, Variance_of_feature2_for_digit1, Accuracy_for_digit0testset, Accuracy_for_digit1testset]


    # each of the matrices in the train0 and train1 has 28 elements

    # print([len(train0),len(train1),len(test0),len(test1)])

    # print('Your trainset and testset are generated successfully!')
    pass


# NB classifier formula
def NB(test, avg, omega):
    exponent = -0.5 * (((test - avg) / omega) ** 2)
    e = math.e ** (exponent)
    denominator = omega * math.sqrt(2 * math.pi)
    return e / denominator


if __name__ == '__main__':
    main()