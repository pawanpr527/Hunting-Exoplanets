import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class TessExoplanetClassifier:
    def __init__(self, file_path, n_estimators=200, random_state=42,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", test_size=0.2,
                 auto_run=True):
        self.file_path = file_path
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.test_size = test_size
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.num_imputer = None
        self.label_encoder_target = None

        if auto_run:
            self.run_pipeline()

    def run_pipeline(self):
        self.load_data()
        self.preprocess()
        self.split_data()
        self.train_model()
        self.evaluate()

    def load_data(self):
        df = pd.read_csv(self.file_path)
        features = [
            # target
            'tfopwg_disp',

            # numeric planetary features
            'pl_orbper', 'pl_trandurh', 'pl_trandeperr1', 'pl_trandeperr2',
            'pl_trandep', 'pl_rade', 'pl_radeerr1', 'pl_radeerr2',
            'pl_insol', 'pl_eqt',

            # stellar properties
            'st_teff', 'st_tefferr1', 'st_tefferr2',
            'st_logg', 'st_rad', 'st_raderr1', 'st_raderr2',
            'st_dist', 'st_disterr1', 'st_disterr2',

            'st_tmag', 'st_tmagerr1', 'st_tmagerr2',

            # proper motions
            'st_pmra', 'st_pmdec',

            # identifiers / categorical
            'toi', 'tid'
        ]
        self.data = df[features].dropna(subset=['tfopwg_disp'])

    def preprocess(self):
        numeric_features = [
            'pl_orbper', 'pl_trandurh', 'pl_trandeperr1', 'pl_trandeperr2',
            'pl_trandep', 'pl_rade', 'pl_radeerr1', 'pl_radeerr2',
            'pl_insol', 'pl_eqt',
            'st_teff', 'st_tefferr1', 'st_tefferr2',
            'st_logg', 'st_rad', 'st_raderr1', 'st_raderr2',
            'st_dist', 'st_disterr1', 'st_disterr2',
            'st_tmag', 'st_tmagerr1', 'st_tmagerr2',
            'st_pmra', 'st_pmdec'
        ]

        categorical_features = ['toi', 'tid']

        # Impute numeric features
        self.num_imputer = SimpleImputer(strategy='median')
        self.data[numeric_features] = self.num_imputer.fit_transform(self.data[numeric_features])

        # Encode categorical features
        for col in categorical_features:
            self.data[col] = self.data[col].fillna('unknown')
            self.data[col] = LabelEncoder().fit_transform(self.data[col].astype(str))

        # Encode target
        self.label_encoder_target = LabelEncoder()
        self.y = self.label_encoder_target.fit_transform(self.data['tfopwg_disp'])

        # Final features
        self.X = self.data[numeric_features + categorical_features]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

    def train_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", report)
        return accuracy, report
    