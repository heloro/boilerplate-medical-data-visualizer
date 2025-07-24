import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")


# 2

df['overweight'] = np.where((df['weight']/((df['height']/100)**2)>25),1,0)

# 3

df['cholesterol'] = np.where(df['cholesterol'] > 1,1,0)
df['gluc'] = np.where(df['gluc'] > 1,1,0)

# 4
def draw_cat_plot():
  #sns.barplot(data = df, x = 'variable', y = 'total')
  
    # 5
    df_cat = pd.melt(df, id_vars = 'cardio', value_vars = ['cholesterol','gluc','smoke','alco','active','overweight'])

    # 6
    df_cat = pd.DataFrame(df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total'))

    # 7
    graph = sns.catplot(data = df_cat, kind = 'bar', x = 'variable', y = 'total', hue = 'value', col = 'cardio')

    # 8
    fig = graph.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    conditions = (
      (df['ap_lo'] <= df['ap_hi']) & 
      (df['height'] >= df['height'].quantile(0.025)) & 
      (df['height'] <= df['height'].quantile(0.975)) &
      (df['weight'] >= df['weight'].quantile(0.025)) &
      (df['weight'] <= df['weight'].quantile(0.975))
      )
      
    df_heat = df.loc[conditions]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr))

    # 14
    fig, ax = plt.subplots(figsize = (16,9))

    # 15
    graph2 = sns.heatmap(corr, mask = mask, square = True, linewidths = 0.5, annot = True, fmt = "0.1f")

    # 16
    fig.savefig('heatmap.png')
    return fig
