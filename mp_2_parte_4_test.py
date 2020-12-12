import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sb
import matplotlib.pyplot as plt
import sys
from openpyxl import load_workbook
from sklearn.preprocessing import LabelEncoder


def main():
    data_route = "./Datasets/completo_test_synth_dengue.csv"  # Default
    if len(sys.argv) > 1:
        data_route = sys.argv[1]  # Dataset passed by argument
    distribution = "Categorical"
    if len(sys.argv) > 2:
        distribution = sys.argv[2]
    model_route = "./TrainedModels/CompleteNB.pckl"
    if len(sys.argv) > 2:
        model_route = sys.argv[2]
    sheet = "Complete"
    if len(sys.argv) > 3:
        sheet = sys.argv[3]

    dataset = pd.read_csv(data_route)
    preProcess(dataset)

    if distribution.lower() == "gaussian":
        testing_data = transformFeaturesBernoulli(dataset)
        cont = dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]]
        testing_data = pd.concat([cont, testing_data], axis=1)
    elif distribution.lower() == "categorical":
        training_data = transformFeaturesBernoulli(dataset)
        cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
        testing_data = pd.concat([cont, training_data], axis=1)
    elif distribution.lower() == "bernoulli":
        training_data = transformFeaturesBernoulli(dataset)
        cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
        testing_data = pd.concat([cont, training_data], axis=1)
    testing_labels = encodeLabels(dataset['clase'])

    model = pd.read_pickle(model_route)
    test_res, test_confm = testModel(testing_data, testing_labels, model)
    printConfusionMatrix(test_confm)
    saveResults(test_res, f'{sheet}NB')


def printConfusionMatrix(confm):
    labels = ['Grave', 'No Signos', 'Alarma', 'No Dengue']
    plt.title("Validation Sets Confusion Matrix", fontsize=18)
    heatmap = sb.heatmap(confm, annot=True, cbar=False, cmap='Blues', fmt='', xticklabels=labels, yticklabels=labels)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30)
    plt.xlabel("True Label", fontsize=16)
    plt.ylabel("Predicted Label", fontsize=16)
    plt.show()


def encodeLabels(classes):
    le = LabelEncoder()
    le.fit(classes)
    labels = {'clase': le.transform(classes)}
    return pd.DataFrame(labels)


# Preprocess data to fix instances of 'NO' and NaN in the dataframe
def preProcess(dataset):
    dataset.replace(np.nan, 'NA', regex=True, inplace=True)
    dataset.replace('NO', 'No', regex=True, inplace=True)


def transformFeaturesBernoulli(dataset):
    exclude = [
        "plaquetas",
        "leucocitos",
        "linfocitos",
        "hematocritos",
    ]
    for i, col in enumerate(dataset.columns):
        if col not in exclude:
            dummified = pd.get_dummies(dataset[col])
            if i > 1:
                ret_dataset = pd.concat([ret_dataset, dummified], axis=1)
            else:
                ret_dataset = dummified

    return ret_dataset


def continuousTransform(dataset):
    cont = dataset['plaquetas'] > 200000
    cont = pd.concat([cont, dataset['leucocitos'] > 7500], axis=1)
    cont = pd.concat([cont, dataset['hematocritos'] > 0.445], axis=1)
    cont = pd.concat([cont, dataset['linfocitos'] > 0.44], axis=1)
    for i, col in enumerate(cont.columns):
        dummified = pd.get_dummies(cont[col])
        if i > 0:
            retVal = pd.concat([retVal, dummified], axis=1)
        else:
            retVal = dummified
    return retVal


def testModel(testing_data, testing_labels, model):
    predicted = model.predict(testing_data)
    results = {}
    results['f1_score'] = [metrics.f1_score(y_true=testing_labels, y_pred=predicted, average='macro')]
    results['accuracy'] = [metrics.accuracy_score(y_true=testing_labels, y_pred=predicted)]
    confm = metrics.confusion_matrix(y_true=testing_labels, y_pred=predicted)
    return pd.DataFrame(results), confm


def saveResults(conf_results, sheet):
    conf_route = './Configurations/TestResults.xlsx'
    book = load_workbook(conf_route)
    writer = pd.ExcelWriter(conf_route, engine='openpyxl')
    writer.book = book
    conf_results.to_excel(excel_writer=writer, sheet_name=sheet)
    writer.save()
    writer.close()


if __name__ == "__main__":
    main()