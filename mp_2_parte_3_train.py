import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import seaborn as sb
import matplotlib.pyplot as plt
import sys
from openpyxl import load_workbook
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    data_route = "./Datasets/completo_train_synth_dengue.csv"  # Default
    if len(sys.argv) > 1:
        data_route = sys.argv[1]  # Dataset passed by argument
    model_route = "./TrainedModels/CompleteSVM.pckl"
    if len(sys.argv) > 2:
        model_route = sys.argv[2]
    conf_route = "./Configurations/Best.xlsx"  # Default
    if len(sys.argv) > 3:
        conf_route = sys.argv[3]  # Configuration file
    sheet = "Complete"
    if len(sys.argv) > 4:
        sheet = sys.argv[4]
    dataset = pd.read_csv(data_route)

    configuration = pd.read_excel(conf_route, sheet_name=f'{sheet}SVM')

    preProcess(dataset)

    training_data = dummify(dataset.loc[:, dataset.columns != 'clase'])
    training_labels = encodeLabels(dataset['clase'])

    val_res, val_confm = runBestConfiguration(training_data, training_labels, configuration)
    printConfusionMatrix(val_confm)
    saveResults(val_res, f'{sheet}SVM')

    model = trainWithAllData(training_data, training_labels, configuration)
    pd.to_pickle(model, model_route)
    print(f'Model saved in {model_route}')


def trainWithAllData(training_data, training_labels, configuration):
    c = configuration.loc[0, 'C']
    kernel = configuration.loc[0, 'Kernel']
    gamma = configuration.loc[0, 'Gamma']
    svc = SVC(
        kernel=kernel,
        C=c,
        gamma=gamma
    )
    model = trainClassifier(training_data, training_labels, svc)
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
        "hematocritos",
    ]
    ret_dataset = dataset[[col for col in exclude if col in dataset.columns]]
    for col in dataset.columns:
        if col not in exclude:
            dummified = pd.get_dummies(dataset[col])
            ret_dataset = pd.concat([ret_dataset, dummified], axis=1)
    return ret_dataset


def encodeLabels(classes):
    le = LabelEncoder()
    le.fit(classes)
    labels = {'clase': le.transform(classes)}
    return pd.DataFrame(labels)


def runBestConfiguration(data, labels, configuration):
    f1_valdict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    for i, conf in configuration.iterrows():
        print(f"Running Iteration #{i+1}...")
        val_f1, val_confm = SVMCrossValidate(
            data,
            labels,
            C=conf['C'],
            kernel=conf['Kernel'],
            gamma=conf['Gamma']
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


def SVMCrossValidate(dataset, labels, kernel='rbf', C=1.0, gamma='auto', k_sets=5, print_res=False):
    kf = KFold(n_splits=k_sets, shuffle=True, random_state=1)
    kf.get_n_splits(dataset)

    f1_valdict = {}
    index = 1

    dimension = len(np.unique(labels))
    confm = np.zeros((dimension, dimension), dtype=int)

    avgval_f1 = 0
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset.loc[train_index], dataset.loc[test_index]
        Y_train, Y_test = labels.loc[train_index], labels.loc[test_index]
        classifier = trainClassifier(X_train, Y_train, SVC(kernel=kernel, C=C, gamma=gamma, random_state=0))
        f1_valdict[f"P{index}"], temp_confm = evaluate(X_test, Y_test, classifier)
        avgval_f1 += f1_valdict[f"P{index}"]
        confm += temp_confm
        index += 1
    avgval_f1 /= k_sets
    if print_res:
        print(f"Validation F1: {avgval_f1}")
    f1_valdict["Promedio"] = avgval_f1
    return f1_valdict, confm


def trainClassifier(X_train, Y_train, svc):
    model = make_pipeline(StandardScaler(), svc)
    model.fit(X_train, Y_train)
    return model


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
    print(f'Results saved in {conf_route}, sheet {sheet}')


if __name__ == "__main__":
    main()