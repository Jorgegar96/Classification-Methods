import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import seaborn as sb
import matplotlib.pyplot as plt
import sys
from openpyxl import load_workbook


def main():
    data_route = "./Datasets/completo_train_synth_dengue.csv"  # Default
    if len(sys.argv) > 1:
        data_route = sys.argv[1]  # Dataset passed by argument
    model_route = "./TrainedModels/CompleteRFC.pckl"
    if len(sys.argv) > 2:
        model_route = sys.argv[2]
    conf_route = "./Configurations/Best.xlsx"  # Default
    if len(sys.argv) > 3:
        conf_route = sys.argv[3]  # Configuration file
    sheet = "Complete"
    if len(sys.argv) > 4:
        sheet = sys.argv[4]
    dataset = pd.read_csv(data_route)

    configuration = pd.read_excel(conf_route, sheet_name=f'{sheet}RFC')

    preProcess(dataset)

    training_data = dummify(dataset.loc[:, dataset.columns != 'clase'])
    training_labels = dataset['clase']

    val_res, val_confm = runBestConfiguration(training_data, training_labels, configuration)
    printConfusionMatrix(val_confm)
    saveResults(val_res, f'{sheet}RFC')

    model = trainWithAllData(training_data, training_labels, configuration)
    pd.to_pickle(model, model_route)


def trainWithAllData(training_data, training_labels, configuration):
    criterion = configuration.loc[0, 'Criterion']
    max_depth = configuration.loc[0, 'max_depth']
    n_estimators = configuration.loc[0, 'n_estimators']
    max_features = configuration.loc[0, 'max_features']
    model =  RandomForestClassifier(
        criterion=criterion,
        max_depth=max_depth,
        n_estimators=n_estimators,
        max_features=max_features
    )
    model.fit(training_data, training_labels)
    return model


def printConfusionMatrix(confm):
    labels = ['Grave', 'No Signos', 'Alarma', 'No Dengue']
    plt.title("Validation Sets Confusion Matrix", fontsize=18)
    heatmap = sb.heatmap(confm, annot=True, cbar=False, cmap='Blues', fmt='', xticklabels=labels, yticklabels=labels)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30)
    plt.xlabel("True Label", fontsize=16)
    plt.ylabel("Predicted Label", fontsize=16)
    plt.show()


# Preprocess data to fix instances of 'NO' and NaN in the dataframe
def preProcess(dataset):
    dataset.replace(np.nan, 'NA', regex=True, inplace=True)
    dataset.replace('NO', 'No', regex=True, inplace=True)


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


def runBestConfiguration(data, labels, configuration):
    f1_valdict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    for i, conf in configuration.iterrows():
        print(f"Running Iteration #{i+1}...")
        val_f1, val_confm = RFCrossValidate(
            data,
            labels,
            max_depth=conf['max_depth'],
            criterion=conf['Criterion'],
            n_estimators=conf['n_estimators'],
            max_features=conf['max_features']
        )
        for key in f1_valdict:
            f1_valdict[key].append(val_f1[key])
    f1_valdict["Accuracy"] = getAccuracy(val_confm)
    val_res = pd.DataFrame(data=f1_valdict)
    return val_res, val_confm


def getAccuracy(confm):
    correct = 0
    for i in range(len(confm)):
        correct += confm[i][i]
    total = np.sum(confm)
    return correct / total


def RFCrossValidate(dataset, labels, max_depth=15, n_estimators=150, criterion='gini',
                    max_features=21, k_sets=5, print_res=False):
    kf = KFold(n_splits=k_sets, shuffle=True, random_state=1)
    kf.get_n_splits(dataset)

    f1_valdict = {}
    val_confm = {}
    index = 1

    dimension = len(np.unique(labels))
    confm = np.zeros((dimension, dimension), dtype=int)

    avgval_f1 = 0
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset.loc[train_index], dataset.loc[test_index]
        Y_train, Y_test = labels.loc[train_index], labels.loc[test_index]
        forest = RandomForestClassifier(
            max_depth=max_depth,
            random_state=0,
            n_estimators=n_estimators,
            criterion=criterion,
            max_features=max_features
        )
        forest.fit(X_train, Y_train)
        f1_valdict[f"P{index}"], val_confm[f"cm{index+1}"] = evaluate(X_test, Y_test, forest)
        avgval_f1 += f1_valdict[f"P{index}"]
        confm += val_confm[f"cm{index+1}"]
        index += 1
    avgval_f1 /= k_sets
    if print_res:
        print(f"Validation F1: {avgval_f1}")
    f1_valdict["Promedio"] = avgval_f1
    return f1_valdict, confm


def evaluate(X_test, Y_test, classifier, print_res=False):
    pred_test = classifier.predict(X_test)
    f1_test = metrics.f1_score(pred_test, Y_test, average="macro")
    val_confm = metrics.confusion_matrix(pred_test, Y_test)
    if print_res:
        printResults(f1_test, val_confm)
    return f1_test, np.array(val_confm)


def printResults(f1_test, val_confm):
    print("-" * 30)
    print(f"Testing Acc: {f1_test}")
    print(val_confm)


def saveResults(conf_results, sheet):
    conf_route = './Configurations/ConfigurationResults.xlsx'
    book = load_workbook(conf_route)
    writer = pd.ExcelWriter(conf_route, engine='openpyxl')
    writer.book = book
    conf_results.to_excel(excel_writer=writer, sheet_name=sheet)
    writer.save()
    writer.close()



if __name__ == "__main__":
    main()