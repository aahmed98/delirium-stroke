import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data_path = "data\day_to_day.csv" # path to CSV file you want to 
columns = ["dynamic_time_warping_two_day_pim_morning","dynamic_time_warping_two_day_pim_afternoon",
         "dynamic_time_warping_two_day_pim_night","dynamic_time_warping_two_day_pim_1_to_1"]  # each element corresponds to one corner
columns.append("delirium_stat")
x_label = "dtw" # x label of plot


df = pd.read_csv(data_path)
filt_df = df[columns]
transformed_data = MinMaxScaler().fit_transform(X=filt_df)
filt_df.loc[:,columns] = transformed_data


plt.subplots(2, 2)
for i, col in enumerate(filt_df.columns[:4]):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(filt_df.loc[filt_df['delirium_stat'] == 1, col], shade=True, label='Delirious')
    sns.kdeplot(filt_df.loc[filt_df['delirium_stat'] == 0, col], shade=True, label='Non-delirious')
    plt.xlabel(x_label)
    plt.title(col)
    if i == 1:
        plt.legend(loc='upper right')
    else:
        plt.legend().remove()

plt.subplot_tool()
plt.show()