import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if not os.path.exists(".\\eval_plots"):
    os.makedirs(".\\eval_plots")

with open('eval_results.pkl', 'rb') as f:
    df = pd.read_pickle(f)

# organize trials s.t. one ind var, other constant
var_vals = {
    'learning_rate': df['learning_rate'].unique(),
    'epsilon_decay': df['epsilon_decay'].unique(),
    'discount_factor': df['discount_factor'].unique()
}

vars = ['learning_rate', 'epsilon_decay', 'discount_factor']
for var in vars:
    static_vars = list(set(vars).difference(set([var])))
    for static_var_1_val in var_vals[static_vars[0]]:
        for static_var_2_val in var_vals[static_vars[1]]:
            df_view = df[(df[static_vars[0]] == static_var_1_val) & (df[static_vars[1]] == static_var_2_val)][[var, 'num_steps']]

            sns.boxplot(x=var, y='num_steps', data=df_view, showfliers=False)
            plt.title(f"Ind Var: {var}, Static Vars: {static_vars[0]} {static_var_1_val}, {static_vars[1]} {static_var_2_val}")
            plt.xlabel(f"Ind Var: {var}")
            plt.ylabel("Number of Steps to Transport Passenger")
            plt.show()

            save_path = f".\\eval_plots\\IndVar {var} Static Vars {static_vars[0]} {static_var_1_val} {static_vars[1]} {static_var_2_val}.png".replace(" ", "_")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
