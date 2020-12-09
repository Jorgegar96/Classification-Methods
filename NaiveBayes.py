import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.model_selection import KFold
from sklearn import metrics
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from openpyxl import load_workbook

def main():
    data_route = "./Datasets/completo_train_synth_dengue.csv"  # Default
    if len(sys.argv) > 1:
        data_route = sys.argv[1]  # Dataset passed by argument
    sheet = "Complete"
    if len(sys.argv) > 2:
        sheet = sys.argv[2]

    dataset = pd.read_csv(data_route)

    runConfigurations(dataset.loc[:, dataset.columns != 'clase'], dataset['clase'], sheet)


def transformFeatures(dataset):
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
    cont = pd.concat([cont, dataset['leucocitos'] > 7000], axis=1)
    cont = pd.concat([cont, dataset['hematocritos'] > 0.440], axis=1)
    cont = pd.concat([cont, dataset['linfocitos'] > 0.42], axis=1)
    for i, col in enumerate(cont.columns):
        dummified = pd.get_dummies(dataset[col])
        if i > 0:
            retVal = pd.concat([retVal, dummified], axis=1)
        else:
            retVal = dummified
    return retVal


def encodeLabels(classes):
    le = LabelEncoder()
    le.fit(classes)
    labels = {'clase': le.transform(classes)}
    return pd.DataFrame(labels)


def runConfigurations(data, labels, sheet):
    f1_traindict = {"Conf_id": [1, 2, 3], 'Dataset': [sheet, sheet, sheet],
                    "Base": ['Gaussian', 'Categorical', 'Bernoulli'],
                    "P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    f1_valdict = {"Conf_id": [1, 2, 4], 'Dataset': [sheet, sheet, sheet],
                  "Base": ['Gaussian', 'Categorical', 'Bernoulli'],
                  "P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}

    train_f1, val_f1 = Gaussian(GaussianNB(), data, labels)
    for key in f1_valdict:
        if key != 'Conf_id' and key != 'Dataset' and key != 'Base':
            f1_traindict[key].append(train_f1[key])
            f1_valdict[key].append(val_f1[key])

    train_f1, val_f1 = Categorical(CategoricalNB(), data, labels)
    for key in f1_valdict:
        if key != 'Conf_id' and key != 'Dataset' and key != 'Base':
            f1_traindict[key].append(train_f1[key])
            f1_valdict[key].append(val_f1[key])

    train_f1, val_f1 = Bernoulli(BernoulliNB(), data, labels)
    for key in f1_valdict:
        if key != 'Conf_id' and key != 'Dataset' and key != 'Base':
            f1_traindict[key].append(train_f1[key])
            f1_valdict[key].append(val_f1[key])

    train_res = pd.DataFrame(data=f1_traindict)
    val_res = pd.DataFrame(data=f1_valdict)

    saveResults(val_res, f"{sheet}NB")
    return train_res, val_res


def Gaussian(model, dataset, labels):
    training_data = transformFeatures(dataset)
    cont = dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]]
    training_data = pd.concat([cont, training_data], axis=1)
    training_labels = encodeLabels(labels)
    return NaiveBayesCrossValidate(model, training_data, training_labels)


def Categorical(model, dataset, labels):
    training_data = transformFeatures(dataset)
    training_labels = encodeLabels(labels)
    return NaiveBayesCrossValidate(model, training_data, training_labels)


def Bernoulli(model, dataset, labels):
    training_data = transformFeatures(dataset)
    cont = continuousTransform(dataset[["plaquetas", "leucocitos", "linfocitos", "hematocritos"]])
    training_data = pd.concat([cont, training_data], axis=1)
    training_labels = encodeLabels(labels)
    return NaiveBayesCrossValidate(model, training_data, training_labels)


def NaiveBayesCrossValidate(model, dataset, labels, k_sets=5, print_res=False):
    kf = KFold(n_splits=k_sets, shuffle=True, random_state=1)
    kf.get_n_splits(dataset)

    f1_traindict, f1_valdict = {}, {}
    index = 1
    avgtrain_f1 = 0
    avgval_f1 = 0
    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset.loc[train_index], dataset.loc[test_index]
        Y_train, Y_test = labels.loc[train_index], labels.loc[test_index]
        model.fit(X_train, Y_train)
        f1_traindict[f"P{index}"], f1_valdict[f"P{index}"] = evaluate(X_train, Y_train, X_test, Y_test, model)
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