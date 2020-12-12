import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from openpyxl import load_workbook
import sys
from sklearn.preprocessing import LabelEncoder


def main():
    data_route = "./Datasets/completo_train_synth_dengue.csv"  # Default
    if len(sys.argv) > 1:
        data_route = sys.argv[1]  # Dataset passed by argument
    distribution = "Categorical"
    if len(sys.argv) > 2:
        distribution = sys.argv[2]
    model_route = "./TrainedModels/CompleteNB.pckl"
    if len(sys.argv) > 3:
        model_route = sys.argv[3]
    sheet = "Complete"
    if len(sys.argv) > 4:
        sheet = sys.argv[4]

    training_data = pd.read_csv(data_route)
    preProcess(training_data)
    val_res, val_confm = runConfiguration(
        training_data.loc[:, training_data.columns != 'clase'], training_data['clase'], distribution
    )
    saveResults(val_res, f"{sheet}NB")
    printConfusionMatrix(val_confm)

    model = trainWithAllData(training_data, training_data['clase'], distribution)
    pd.to_pickle(model, model_route)


# Preprocess data to fix instances of 'NO' and NaN in the dataframe
def preProcess(dataset):
    dataset.replace(np.nan, 'NA', regex=True, inplace=True)
    dataset.replace('NO', 'No', regex=True, inplace=True)


def trainWithAllData(dataset, labels, distribution):
    if distribution.lower() == "gaussian":
        training_data = transformFeaturesBernoulli(dataset)
        cont = dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]]
        training_data = pd.concat([cont, training_data], axis=1)
        model = make_pipeline(StandardScaler(), GaussianNB())
    elif distribution.lower() == "categorical":
        training_data = transformFeaturesBernoulli(dataset)
        cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
        training_data = pd.concat([cont, training_data], axis=1)
        model = CategoricalNB()
    elif distribution.lower() == "bernoulli":
        training_data = transformFeaturesBernoulli(dataset)
        cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
        training_data = pd.concat([cont, training_data], axis=1)
        model = BernoulliNB
    training_labels = encodeLabels(labels)
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


def runConfiguration(data, labels, distribution):
    f1_valdict = {"Base": [distribution],"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}

    if distribution.lower() == "gaussian":
        val_f1, val_confm = Gaussian(GaussianNB(), data, labels)
        for key in f1_valdict:
            if key != 'Base':
                f1_valdict[key].append(val_f1[key])
    elif distribution.lower() == "categorical":
        val_f1, val_confm = Categorical(CategoricalNB(), data, labels)
        for key in f1_valdict:
            if key != 'Base':
                f1_valdict[key].append(val_f1[key])
    elif distribution.lower() == "bernoulli":
        val_f1, val_confm = Bernoulli(BernoulliNB(), data, labels)
        for key in f1_valdict:
            if key != 'Base':
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


def Gaussian(model, dataset, labels):
    training_data = transformFeaturesBernoulli(dataset)
    cont = dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]]
    training_data = pd.concat([cont, training_data], axis=1)
    training_labels = encodeLabels(labels)
    classifier = make_pipeline(StandardScaler(), model)
    return NaiveBayesCrossValidate(classifier, training_data, training_labels)


def Categorical(model, dataset, labels):
    training_data = transformFeaturesBernoulli(dataset)
    cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
    training_data = pd.concat([cont, training_data], axis=1)
    training_labels = encodeLabels(labels)
    return NaiveBayesCrossValidate(model, training_data, training_labels)


def Bernoulli(model, dataset, labels):
    training_data = transformFeaturesBernoulli(dataset)
    cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
    training_data = pd.concat([cont, training_data], axis=1)
    training_labels = encodeLabels(labels)
    return NaiveBayesCrossValidate(model, training_data, training_labels)


def NaiveBayesCrossValidate(model, dataset, labels, k_sets=5, print_res=False):
    kf = KFold(n_splits=k_sets, shuffle=True, random_state=1)
    kf.get_n_splits(dataset)

    f1_valdict = {}

    dimension = len(np.unique(labels))
    confm = np.zeros((dimension, dimension), dtype=int)

    index = 1
    avgval_f1 = 0
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset.loc[train_index], dataset.loc[test_index]
        Y_train, Y_test = labels.loc[train_index], labels.loc[test_index]
        model.fit(X_train, Y_train)
        f1_valdict[f"P{index}"], temp_confm = evaluate(X_test, Y_test, model)
        avgval_f1 += f1_valdict[f"P{index}"]
        confm += temp_confm
        index += 1

    avgval_f1 /= k_sets
    if print_res:
        print(f"Testing F1: {avgval_f1}")

    f1_valdict["Promedio"] = avgval_f1
    return f1_valdict, confm


def evaluate(X_test, Y_test, classifier, print_res=False):
    pred_test = classifier.predict(X_test)
    f1_test = metrics.f1_score(pred_test, Y_test, average="macro")
    val_confm = metrics.confusion_matrix(pred_test, Y_test)
    if print_res:
        printResults(f1_test,  val_confm)
    return f1_test, val_confm


def printResults(f1_test, val_confm):
    print("-" * 30)
    print(f"Testing Acc: {f1_test}")
    print(val_confm)


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


def saveResults(conf_results, sheet):
    conf_route = './Configurations/ConfigurationResults.xlsx'
    book = load_workbook(conf_route)
    writer = pd.ExcelWriter(conf_route, engine='openpyxl')
    writer.book = book
    conf_results.to_excel(excel_writer=writer, sheet_name=sheet)
    writer.save()
    writer.close()


def encodeLabels(classes):
    le = LabelEncoder()
    le.fit(classes)
    labels = {'clase': le.transform(classes)}
    return pd.DataFrame(labels)


if __name__ == "__main__":
    main()