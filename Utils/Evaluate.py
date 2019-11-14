import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


def confusionMatrixDF(true_y, pred_y, labels):
    
    df = pd.DataFrame(confusion_matrix(true_y, pred_y), 
                     columns=labels,
                     index=labels)
    return df

def evaluateModel(model, testX, testY, labelNames, batchSize=64):
    # Evaluate network performance
    print('[INFO] evaluating network performance...')
    
    predictions = model.predict(testX, batch_size=batchSize)
    
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames))
    
    print('[INFO] Confusion Matrix....')
    confMat = confusionMatrixDF(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                labels=labelNames)
    return confMat