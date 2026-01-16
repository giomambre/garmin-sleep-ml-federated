from sklearn.ensemble import RandomForestRegressor
from config import N_TREES_PER_CLIENT, MAX_DEPTH, MIN_SAMPLES_LEAF, SEED

class LocalClient:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = RandomForestRegressor(
            n_estimators=N_TREES_PER_CLIENT,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=SEED,
            n_jobs=-1
        )

    def train(self):
        self.model.fit(self.X, self.y)
        return self.model.estimators_