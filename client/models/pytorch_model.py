from sklearn.linear_model import Perceptron

def create_seed_model(exp_config):
	model = Perceptron(max_iter=1, warm_start=True)
	return model
