import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
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
    model_route = "./TrainedModels/CompleteRFC.pckl"
    if len(sys.argv) > 2:
        model_route = sys.argv[1]
    sheet = "Complete"
    if len(sys.argv) > 3:
        sheet = sys.argv[2]
    res_route = "./Configurations/TestResults.xlsx"  # Default
    if len(sys.argv) > 4:
        conf_route = sys.argv[3]  # Configuration file
    dataset = pd.read_csv(data_route)

    testing_data = dummify(dataset.loc[:, dataset.columns != 'clase'])
    testing_labels = dataset['clase']

    model = pd.read_pickle(model_route)
    test_res, test_confm = testModel(testing_data, testing_labels, model)
    printConfusionMatrix(test_confm)
    saveResults(test_res, f'{sheet}RFC')


def printConfusionMatrix(confm):
    labels = ['Grave', 'No Signos', 'Alarma', 'No Dengue']
    plt.title("Validation Sets Confusion Matrix", fontsize=18)
    heatmap = sb.heatmap(confm, annot=True, cbar=False, cmap='Blues', fmt='', xticklabels=labels, yticklabels=labels)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30)
    plt.xlabel("True Label", fontsize=16)
    plt.ylabel("Predicted Label", fontsize=16)
    plt.show()


def dummify(dataset):
    exclude = [
        "plaquetas",
        "leucocitos",
        "linfocitos",
        "hematocritos"
    ]
    ret_dataset = dataset[[col for col in exclude if col in dataset.columns]]
    for col in dataset.columns:
        if col not in exclude:
            dummified = pd.get_dummies(dataset[col])
            ret_dataset = pd.concat([ret_dataset, dummified], axis=1)
    return ret_dataset


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