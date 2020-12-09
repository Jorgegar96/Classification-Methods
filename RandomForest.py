import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import sys
#import xlrd  # Needed to be installed even though not imported to run
from openpyxl import load_workbook


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
    configurations = pd.read_excel(conf_route, sheet_name=f'{sheet}RFC')

    training_data = dummify(dataset.loc[:, dataset.columns != 'clase'])
    training_labels = dataset['clase']

    _, val_res = runConfigurations(training_data, training_labels, configurations)
    conf_results = pd.concat([configurations, val_res], axis=1)
    saveResults(conf_results, f'{sheet}RFC')


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


def runConfigurations(data, labels, configurations):
    f1_traindict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    f1_valdict = {"P1": [], "P2": [], "P3": [], "P4": [], "P5": [], "Promedio": []}
    for i, conf in configurations.iterrows():
        print(f"Running Iteration #{i+1}...")
        train_f1, val_f1 = RFCrossValidate(
            data,
            labels,
            max_depth=conf['max_depth'],
            criterion=conf['Criterion'],
            n_estimators=conf['n_estimators'],
            max_features=conf['max_features']
        )
        for key in f1_valdict:
            f1_traindict[key].append(train_f1[key])
            f1_valdict[key].append(val_f1[key])

    train_res = pd.DataFrame(data=f1_traindict)
    val_res = pd.DataFrame(data=f1_valdict)
    return train_res, val_res


def RFCrossValidate(dataset, labels, max_depth=15, n_estimators=150, criterion='gini',
                    max_features=21, k_sets=5, print_res=False):
    kf = KFold(n_splits=k_sets, shuffle=True, random_state=1)
    kf.get_n_splits(dataset)

    f1_traindict, f1_valdict = {}, {}
    train_confm, val_confm = {}, {}
    index = 1

    avgtrain_f1 = 0
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
        predicted = forest.predict(X_train)
        f1_traindict[f"P{index}"] = metrics.f1_score(predicted, Y_train, average="macro")
        avgtrain_f1 += f1_traindict[f"P{index}"]
        train_confm[f"cm{index+1}"] = metrics.confusion_matrix(predicted, Y_train)
        if print_res:
            print("-"*30)
            print(f"Training Acc: {metrics.accuracy_score(predicted, Y_train)}")
            print(train_confm[f"cm{index+1}"])
        predicted = forest.predict(X_test)
        f1_valdict[f"P{index}"] = metrics.f1_score(predicted, Y_test, average="macro")
        avgval_f1 += f1_valdict[f"P{index}"]
        val_confm[f"cm{index+1}"] = metrics.confusion_matrix(predicted, Y_test)
        if print_res:
            print(f"Testing Acc: {metrics.accuracy_score(predicted, Y_test)}")
            print(val_confm[f"cm{index+1}"])
        index += 1
    avgtrain_f1 /= k_sets
    avgval_f1 /= k_sets
    if print_res:
        print(f"Training F1: {avgtrain_f1}")
        print(f"Testing F1: {avgval_f1}")
    f1_traindict["Promedio"] = avgtrain_f1
    f1_valdict["Promedio"] = avgval_f1
    return f1_traindict, f1_valdict


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