import pickle

def savePickle(pickleObj, filePath):

    file_Name = filePath
    # open the file for writing
    fileObject = open(file_Name,'wb') 

    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(pickleObj,fileObject)   

    # here we close the fileObject
    fileObject.close()


