import optuna

# 目的関数の定義
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return x ** 2

# GridSamplerのインスタンスを作成
grid_sampler = optuna.samplers.GridSampler({'x': [-10, -5, 0, 5, 10]})

# OptunaのStudyを作成し、GridSamplerを指定して最適化を行う
study = optuna.create_study(sampler=grid_sampler)
study.optimize(objective, n_trials=10)

# 結果の表示
print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}, Params: {trial.params}")
