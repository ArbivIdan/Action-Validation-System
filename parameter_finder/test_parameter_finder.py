import pandas as pd

from ParameterFinder import ParameterFinder

# Read the CSV file
action_enrichment_df = pd.read_csv("../data/action_enrichment_ds_home_exercise.csv")
X = action_enrichment_df["Text"].astype(str)
actions_1 = action_enrichment_df.loc[X.index, "action_1"]
actions_2 = action_enrichment_df.loc[X.index, "action_2"]

y = action_enrichment_df.loc[X.index, "Label"]
X_original = action_enrichment_df.loc[X.index, "Text"]
parameter_finder = ParameterFinder()

for sentence_original, sentence, label, action_1, action_2 in zip(X_original, X, y, actions_1, actions_2):
    found_params = parameter_finder.get_parameters(sentence_original)

    if len(found_params) == 0 or action_1 not in found_params:
        print(sentence_original)
        print(found_params)
        print(action_1)
        print(action_2)
        print("-------")
