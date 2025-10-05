import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
class ExoplanetClassifier:
    def __init__(self, file_path, n_estimators=200, 
                 random_state=42,max_depth=None,min_samples_split=2,min_samples_leaf=1,
                 max_features="sqrt", test_size=0.2, auto_run=True):
        """
        If auto_run=True, the full pipeline (load, preprocess, split, train, evaluate) runs automatically.
        """
        self.file_path = file_path
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.test_size = test_size
        
        self.model = None
        self.label_encoder_target = None
        self.num_imputer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.data = None

        if auto_run:
            self.run_pipeline()

    def run_pipeline(self):
        """Run full pipeline automatically."""
        self.load_data()
        self.preprocess()
        self.split_data()
        self.train_model()
        self.evaluate()

    def load_data(self):
        df = pd.read_csv(self.file_path)
        features = [
            'pl_name', 'hostname', 'default_flag', 'disposition', 'disp_refname',
            'sy_snum', 'sy_pnum', 'discoverymethod', 'disc_year', 'disc_facility',
            'soltype', 'pl_controv_flag', 'pl_refname', 'pl_orbper',
            'pl_rade', 'pl_radj', 'ttv_flag', 'st_refname',
            'st_rad', 'st_teff', 'st_mass', 'st_logg',
            'sy_dist', 'sy_vmag', 'sy_kmag', 'sy_gaiamag'
        ]
        self.data = df[features]

    def preprocess(self):
        numeric_features = [
            'pl_orbper', 'pl_rade', 'pl_radj',
            'st_teff', 'st_rad', 'st_mass', 'st_logg',
            'sy_snum', 'sy_pnum', 'sy_dist',
            'sy_vmag', 'sy_kmag', 'sy_gaiamag'
        ]
        categorical_features = [
            'pl_name', 'hostname', 'disp_refname', 'discoverymethod',
            'disc_facility', 'soltype', 'pl_refname', 'st_refname'
        ]

        # Numeric imputation
        num_imputer = SimpleImputer(strategy='median')
        self.data[numeric_features] = num_imputer.fit_transform(self.data[numeric_features])

        # Categorical encoding
        for col in categorical_features:
            self.data[col] = self.data[col].fillna('unknown')
            self.data[col] = LabelEncoder().fit_transform(self.data[col].astype(str))

        # Target encoding
        self.label_encoder_target = LabelEncoder()
        self.y = self.label_encoder_target.fit_transform(self.data['disposition'])
        self.X = self.data.drop(columns=['disposition'])

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
      self.y_pred = self.model.predict(self.X_test)  # store predictions
      accuracy = accuracy_score(self.y_test, self.y_pred)
      report = classification_report(self.y_test, self.y_pred)
      print("Accuracy:", accuracy)
      print("\nClassification Report:\n", report)
      return accuracy, report


    