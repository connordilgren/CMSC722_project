import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# for evaluating goal skill network vs monolithic agent

with open('eval_results\\eval_results_gsn_1000_eps.pkl', 'rb') as f:
    df_gsn = pd.read_pickle(f)
df_gsn['Source'] = 'GSN 1k Training Eps'
df_gsn_view = df_gsn[(df_gsn['learning_rate'] == 0.1) & (df_gsn['epsilon_decay'] == 0.0002) & (df_gsn['discount_factor'] == 0.99)][['num_steps', 'Source']]

with open('eval_results\\eval_results_monolithic_1000_eps.pkl', 'rb') as f:
    df_mono = pd.read_pickle(f)
df_mono['Source'] = 'Mono 1k Training Eps'
df_mono_view = df_mono[['num_steps', 'Source']]

# with open('eval_results\\eval_results_monolithic_8000_eps.pkl', 'rb') as f:
#     df_mono_20_000 = pd.read_pickle(f)
# df_mono_20_000['Source'] = 'Mono 8k Training Eps'
# df_mono_20_000_view = df_mono_20_000[['num_steps', 'Source']]

df_combined = pd.concat([df_gsn_view, df_mono_view])

sns.boxplot(x='Source', y='num_steps', data=df_combined, showfliers=False)
plt.title(f"Goal-Skill Network Agent vs Monolithic Policy Agent Performance")
plt.xlabel(f"Agent Type")
plt.ylabel("Number of Steps to Transport Passenger")

save_path = ".\\eval_plots\\gsn_vs_mono\\lr_0_1_ed_0_0002_df_0_99_1000_eps.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()


# # for tuning the hyperparameters of SARSA agent

# if not os.path.exists(".\\eval_plots"):
#     os.makedirs(".\\eval_plots")

# with open('eval_results.pkl', 'rb') as f:
#     df = pd.read_pickle(f)

# # organize trials s.t. one ind var, other constant
# var_vals = {
#     'learning_rate': df['learning_rate'].unique(),
#     'epsilon_decay': df['epsilon_decay'].unique(),
#     'discount_factor': df['discount_factor'].unique()
# }

# vars = ['learning_rate', 'epsilon_decay', 'discount_factor']
# for var in vars:
#     static_vars = list(set(vars).difference(set([var])))
#     for static_var_1_val in var_vals[static_vars[0]]:
#         for static_var_2_val in var_vals[static_vars[1]]:
#             df_view = df[(df[static_vars[0]] == static_var_1_val) & (df[static_vars[1]] == static_var_2_val)][[var, 'num_steps']]

#             sns.boxplot(x=var, y='num_steps', data=df_view, showfliers=False)
#             plt.title(f"Ind Var: {var}, Static Vars: {static_vars[0]} {static_var_1_val}, {static_vars[1]} {static_var_2_val}")
#             plt.xlabel(f"Ind Var: {var}")
#             plt.ylabel("Number of Steps to Transport Passenger")
#             plt.show()

#             save_path = f".\\eval_plots\\IndVar {var} Static Vars {static_vars[0]} {static_var_1_val} {static_vars[1]} {static_var_2_val}.png".replace(" ", "_")
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             plt.close()
