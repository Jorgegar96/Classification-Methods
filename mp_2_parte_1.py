import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys
#import xlsxwriter  # Needed to be imported to use xlsxwriter engine


def main():
    route = "./Datasets/completo_train_synth_dengue.csv"  # Default Dataset
    if len(sys.argv) > 1:
        route = sys.argv[1]  # Dataset passed by argument
    dataset = pd.read_csv(route)
    preProcess(dataset)
    generateStatistics(
        dataset,
        contin=[
            'plaquetas',
            'leucocitos',
            'linfocitos',
            'hematocritos'
        ],
        both=[
            'dias_fiebre',
            'dias_ultima_fiebre'
        ],
        print_res=True
    )


# Preprocess data to fix instances of 'NO' and NaN in the dataframe
def preProcess(dataset):
    dataset.replace(np.nan, 'NA', regex=True, inplace=True)
    dataset.replace('NO', 'No', regex=True, inplace=True)


# Generates statistics for each feature in the dataset
def generateStatistics(dataset, contin=None, both=None, print_res=False):
    categ_results = {}  # Stores results for categorical features
    categ = [feature for feature in dataset.columns if feature not in contin]  # List of categorical features
    for feature in categ:
        if feature != 'clase':
            categ_results[feature] = categ_stats(feature, dataset)

    saveCategStats(categ_results)

    if print_res:
        for feature in categ:
            if feature != 'clase':
                print(categ_results[feature].head(10))

    for index, feature in enumerate(contin + both):
        plotContinuous(feature, dataset, index)


# Creates the contingency matrices for each categorical feature
def categ_stats(feature, dataset):
    counts = dataset.groupby([feature, 'clase']).size().reset_index(name="Counts")
    index = np.unique(counts[feature])
    cols = np.unique(counts['clase'])
    init_data = np.zeros((len(index), len(cols)), dtype=int)
    stats = pd.DataFrame(init_data, index=index, columns=cols)
    for clase in cols:
        for valor in index:
            mask = np.logical_and(counts[feature] == valor, counts["clase"] == clase)
            if True in np.array(mask):
                data = counts.loc[mask, "Counts"].values[0]
                stats.at[valor, clase] = data
    stats.index.name = feature
    return stats


# Saves the resulting contingency matrices into an excel file, each in a different sheet
def saveCategStats(stats):
    writer = pd.ExcelWriter('./Data-Analysis/CategoricalStats.xlsx', engine='xlsxwriter')
    for stat in stats:
        stats[stat].to_excel(writer, sheet_name=stats[stat].index.name)
    writer.save()


# Plots BoxPlots for each continuous feature
def plotContinuous(feature, dataset, index):
    plt.figure(figsize=(15, 5))
    sb.set(font_scale=1.5)
    sb.boxplot(y="clase", x=feature, data=dataset)
    plt.savefig(f"./Data-Analysis/BoxPlots/BoxPlot{index}-{feature}.png")
    plt.show()


if __name__ == "__main__":
    main()