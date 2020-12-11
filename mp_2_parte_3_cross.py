import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from openpyxl import load_workbook


# import xlrd  # Needed to be installed even though not imported to run


def main():
    data_route = "./Datasets/completo_train_synth_dengue.csv"  # Default
    if len(sys.argv) > 1:
        data_route = sys.argv[1]  # Dataset passed by argument
    conf_route = "./Configurations/Hyperparameters.xlsx"  # Default
    if len(sys.argv) > 2:
        conf_route = sys.argv[2]  # Configuration file
    sheet = "Complete"
    if len(sys.argv) > 3:
        sheet = sys.argv[3]

    dataset = pd.read_csv(data_route)
    configurations = pd.read_excel(conf_route, sheet_name=f'{sheet}SVM')

    preProcess(dataset)

    training_data = dummify(dataset.loc[:, dataset.columns != 'clase'])
    training_labels = encodeLabels(dataset['clase'])

    _, val_res = runConfigurations(training_data, training_labels, configurations)
    conf_results = pd.concat([configurations, val_res], axis=1)
    saveResults(conf_results, f'{sheet}SVM')


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


def runConfigurations(data, labels, configurations):
    f1_traindict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    f1_valdict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    for i, conf in configurations.iterrows():
        print(f"Running Iteration #{i + 1}...")
        train_f1, val_f1 = SVMCrossValidate(
            data,
            labels,
            kernel=conf['Kernel'],
            C=conf['C'],
            gamma=conf['Gamma']
        )
        for key in f1_valdict:
            f1_traindict[key].append(train_f1[key])
            f1_valdict[key].append(val_f1[key])

    train_res = pd.DataFrame(data=f1_traindict)
    val_res = pd.DataFrame(data=f1_valdict)
    return train_res, val_res


def SVMCrossValidate(dataset, labels, kernel='rbf', C=1.0, gamma='auto', k_sets=5, print_res=False):
    kf = KFold(n_splits=k_sets, shuffle=True, random_state=1)
    kf.get_n_splits(dataset)

    f1_traindict, f1_valdict = {}, {}
    index = 1
    avgtrain_f1 = 0
    avgval_f1 = 0
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset.loc[train_index], dataset.loc[test_index]
        Y_train, Y_test = labels.loc[train_index], labels.loc[test_index]
        classifier = trainClassifier(X_train, Y_train, SVC(kernel=kernel, C=C, gamma=gamma, random_state=0))
        f1_traindict[f"P{index}"], f1_valdict[f"P{index}"] = evaluate(X_train, Y_train, X_test, Y_test, classifier)
        avgtrain_f1 += f1_traindict[f"P{index}"]
        avgval_f1 += f1_valdict[f"P{index}"]
        index += 1

    avgtrain_f1 /= k_sets
    avgval_f1 /= k_sets
    if print_res:
        print(f"Training F1: {avgtrain_f1}")
        print(f"Testing F1: {avgval_f1}")

    f1_traindict["Promedio"] = avgtrain_f1
    f1_valdict["Promedio"] = avgval_f1
    return f1_traindict, f1_valdict


def trainClassifier(X_train, Y_train, svc):
    model = make_pipeline(StandardScaler(), svc)
    model.fit(X_train, Y_train)
    return model


def evaluate(X_train, Y_train, X_test, Y_test, classifier, print_res=False):
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)
    f1_train = metrics.f1_score(pred_train, Y_train, average="macro")
    f1_test = metrics.f1_score(pred_test, Y_test, average="macro")
    train_confm = metrics.confusion_matrix(pred_train, Y_train)
    val_confm = metrics.confusion_matrix(pred_test, Y_test)
    if print_res:
        printResults(f1_test, f1_train, train_confm, val_confm)
    return f1_train, f1_test


def printResults(f1_test, f1_train, train_confm, val_confm):
    print("-" * 30)
    print(f"Training Acc: {f1_train}")
    print(train_confm)
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
