import train_model

#resnet18
train_model.run_experiment(34,1e-4,0.001)
train_model.run_experiment(34,1e-4,0.0001)
train_model.run_experiment(34,1e-4,0.00001)

