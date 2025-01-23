import pandas as pd
import numpy as np

# Provided data
data = [
    "01_1.8048249483108585_100-Site_DKA-M1_A-Phase.csv",
    "02_5.489253242810574_57-Site_DKA-M16_A-Phase.csv",
    "03_4.0456333955129_99-Site_DKA-M4_C-Phase.csv",
    "04_5.5049305359522505_56-Site_DKA-M20_A-Phase.csv",
    "05_4.606877764066067_70-Site_DKA-M5_A-Phase.csv",
    "06_5.5889055331548_59-Site_DKA-M19_C-Phase.csv",
    "07_4.681516726811734_106-Site_DKA-M5_C-Phase.csv",
    "08_5.588966925938926_55-Site_DKA-M20_B-Phase.csv",
    "09_4.713372310002634_74-Site_DKA-M18_C-Phase.csv",
    "10_5.641808271408075_60-Site_DKA-M18_A-Phase.csv",
    "11_4.722724914550775_84-Site_DKA-M5_B-Phase.csv",
    "12_5.648005247116092_64-Site_DKA-M17_B-Phase.csv",
    "13_5.025952617327375_214-Site_DKA-M18_B-Phase_II.csv",
    "14_5.651080489158642_54-Site_DKA-M15_C-Phase.csv",
    "15_5.106038808822617_92-Site_DKA-M6_B-Phase.csv",
    "16_5.675088723500576_212-Site_DKA-M15_C-Phase_II.csv",
    "17_5.1098860502243_58-Site_DKA-M17_C-Phase.csv",
    "18_5.858203013738009_205-Site_Archived_DKA-M15_BPhase_UMG_QCells.csv",
    "19_5.1862111488978_52-Site_DKA-M16_C-Phase.csv",
    "20_6.012480338414508_93-Site_DKA-M4_A-Phase.csv",
    "21_5.252291798591625_63-Site_DKA-M17_A-Phase.csv",
    "22_6.012586116790767_213-Site_DKA-M16_A-Phase_II.csv",
    "23_5.279052972793592_77-Site_DKA-M18_B-Phase.csv",
    "24_6.107803106308_66-Site_DKA-M16_B-Phase.csv",
    "25_5.291588862737025_73-Site_DKA-M19_A-Phase.csv",
    "26_6.494097034136466_79-Site_DKA-M6_A-Phase.csv",
    "27_5.377141674359625_71-Site_DKA-M2_C-Phase.csv",
    "28_6.640705426534017_69-Site_DKA-M4_B-Phase.csv",
    "29_5.381122271219891_218-Site_DKA-M4_C-Phase_II.csv",
    "30_8.239588975906374_67-Site_DKA-M8_A-Phase.csv",
    "31_5.392713745435067_72-Site_DKA-M15_B-Phase.csv",
    "32_8.4429276784261_68-Site_DKA-M8_C-Phase.csv",
    "33_5.424700140953067_90-Site_DKA-M3_A-Phase.csv",
    "34_8.563250303268434_98-Site_DKA-M8_B-Phase.csv",
    "35_5.4838057756424_61-Site_DKA-M15_A-Phase.csv",
    "36_9.772030671437577_85-Site_DKA-M7_A-Phase.csv",
]

# Parse data into DataFrame
def parse_data(data):
    parsed_data = []
    for entry in data:
        parts = entry.split('_')
        index = int(parts[0])
        capacity = float(parts[1])
        parsed_data.append((index, capacity, entry))
    return pd.DataFrame(parsed_data, columns=["Index", "Capacity", "Filename"])

# Balanced split logic
def balanced_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    df = df.sort_values(by="Capacity").reset_index(drop=True)
    bins = np.linspace(df["Capacity"].min(), df["Capacity"].max(), 11)
    df["Bin"] = pd.cut(df["Capacity"], bins=bins, labels=False)

    train, val, test = [], [], []
    for _, group in df.groupby("Bin"):
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        total_count = len(group)
        train_size = int(total_count * train_ratio)
        val_size = int(total_count * val_ratio)

        train.append(group.iloc[:train_size])
        val.append(group.iloc[train_size:train_size + val_size])
        test.append(group.iloc[train_size + val_size:])

    train_df = pd.concat(train).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test).sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, val_df, test_df

# Convert to dictionary
def create_split_dict(train, val, test, site_name="DKASC_AliceSprings"):
    return {
        site_name: {
            "train": train["Index"].tolist(),
            "val": val["Index"].tolist(),
            "test": test["Index"].tolist(),
        }
    }

# Execution
df = parse_data(data)
train_df, val_df, test_df = balanced_split(df)
split_dict = create_split_dict(train_df, val_df, test_df)

# Displaying the final result for verification
import pprint
pprint.pprint(split_dict)