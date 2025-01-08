import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_ecoli_columns():
    return [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "spam"
    ]


def get_beans_columns():
    return [
        "A",
        "P",
        "L",
        "l",
        "K",
        "Ec",
        "C",
        "Ed",
        "Ex",
        "S",
        "R",
        "CO",
        "SF1",
        "SF2",
        "SF3",
        "SF4",
        "beans"
    ]


def get_seeds_columns():
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'seeds']


def get_cancer_columns():
    return ["BI-RADS",
            "Age",
            "Shape",
            "Margin",
            "Density",
            "Severity"]


def get_yeast_columns():

    return ["Sequence Name",
            "mcg",
            "gvh",
            "alm",
            "mit",
            "erl",
            "pox",
            "vac",
            "nuc",
            'target']


def get_covertype_columns():

    return [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",
        "Soil_Type1",
        "Soil_Type2",
        "Soil_Type3",
        "Soil_Type4",
        "Soil_Type5",
        "Soil_Type6",
        "Soil_Type7",
        "Soil_Type8",
        "Soil_Type9",
        "Soil_Type10",
        "Soil_Type11",
        "Soil_Type12",
        "Soil_Type13",
        "Soil_Type14",
        "Soil_Type15",
        "Soil_Type16",
        "Soil_Type17",
        "Soil_Type18",
        "Soil_Type19",
        "Soil_Type20",
        "Soil_Type21",
        "Soil_Type22",
        "Soil_Type23",
        "Soil_Type24",
        "Soil_Type25",
        "Soil_Type26",
        "Soil_Type27",
        "Soil_Type28",
        "Soil_Type29",
        "Soil_Type30",
        "Soil_Type31",
        "Soil_Type32",
        "Soil_Type33",
        "Soil_Type34",
        "Soil_Type35",
        "Soil_Type36",
        "Soil_Type37",
        "Soil_Type38",
        "Soil_Type39",
        "Soil_Type40",
        'Cover_type'
    ]


def get_ozone_columns():

    return [
        "Date",
        "WSR0",
        "WSR1",
        "WSR2",
        "WSR3",
        "WSR4",
        "WSR5",
        "WSR6",
        "WSR7",
        "WSR8",
        "WSR9",
        "WSR10",
        "WSR11",
        "WSR12",
        "WSR13",
        "WSR14",
        "WSR15",
        "WSR16",
        "WSR17",
        "WSR18",
        "WSR19",
        "WSR20",
        "WSR21",
        "WSR22",
        "WSR23",
        "WSR_PK",
        "WSR_AV",
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "T13",
        "T14",
        "T15",
        "T16",
        "T17",
        "T18",
        "T19",
        "T20",
        "T21",
        "T22",
        "T23",
        "T_PK",
        "T_AV",
        "T85",
        "RH85",
        "U85",
        "V85",
        "HT85",
        "T70",
        "RH70",
        "U70",
        "V70",
        "HT70",
        "T50",
        "RH50",
        "U50",
        "V50",
        "HT50",
        "KI",
        "TT",
        "SLP",
        "SLP_",
        "Precp",
        'ozone'
    ]


def read_data(dataset_name):
    if dataset_name == 'ecoli':
        columns = get_ecoli_columns()
        path = "data/spambase/spambase.data"
        dummy_columns = None
        target = "spam"
        convert_target = False
        drop_columns = None
    elif dataset_name == "dry_bean":
        columns = get_beans_columns()
        path = "data/DryBeanDataset/Dry_Bean_Dataset.csv"
        dummy_columns = None
        target = 'beans'
        convert_target = True
        drop_columns = None
    elif dataset_name == 'seeds':
        columns = get_seeds_columns()
        path = "data/seeds/seeds_dataset.csv"
        dummy_columns = None
        target = 'seeds'
        convert_target = True
        drop_columns = None
    elif dataset_name == 'cover_type':
        columns = get_covertype_columns()
        path = "data/covertype/covtype.data"
        dummy_columns = None
        target = "Cover_type"
        convert_target = True
        drop_columns = None
    elif dataset_name == 'ozone':
        columns = get_ozone_columns()
        path = "data/ozone+level+detection/eighthr.csv"
        dummy_columns = None
        target = "ozone"
        convert_target = True
        drop_columns = ['Date']
    elif dataset_name == 'mammographic':
        columns = get_cancer_columns()
        path = "data/mammographic+mass/mammographic_masses.csv"
        target = "Severity"
        dummy_columns = ["Shape", "Margin", "Density", "BI-RADS"]
        convert_target = False
        drop_columns = None
    elif dataset_name == "yeast":
        columns = get_yeast_columns()
        path = "data/yeast/yeast.csv"
        drop_columns = "Sequence Name"
        target = 'target'
        convert_target = True
        dummy_columns = None

    data = pd.read_csv(path).dropna()
    data.columns = columns

    if dummy_columns:
        dummy_dfs = []
        for col in dummy_columns:
            dummy_dfs.append(pd.get_dummies(data[col]).astype(int))

        data = pd.concat([data, *dummy_dfs], axis=1)
        data.drop(columns=dummy_columns, inplace=True)

        bool_columns = data.select_dtypes(['bool']).columns

        data[bool_columns] = data[bool_columns].astype(int)
    if convert_target:
        data[target] = data[target].astype('category').cat.codes.astype(int)

    if drop_columns:
        data.drop(columns=drop_columns, inplace=True)

    return data, target, len(data[target].value_counts()), len(data.columns) - 1


def convert_df_to_tensor(dfs):
    if isinstance(dfs, list):
        return [torch.tensor(df.values, dtype=torch.float32) for df in dfs]
    else:
        return torch.tensor(dfs.values, dtype=torch.float32)


def convert_to_dataloader(X, y):
    tensor_ds = TensorDataset(X, y)
    return DataLoader(tensor_ds)
