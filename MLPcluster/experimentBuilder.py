import train_model

#resnet18
train_model.run_experiment(18,1e-3)
train_model.run_experiment(18,1e-4)
train_model.run_experiment(18,1e-5)

#resnet34
train_model.run_experiment(34,1e-3)
train_model.run_experiment(34,1e-4)
train_model.run_experiment(34,1e-5)