import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class KOIClassifier:
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
        self.label_encoders = {}
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
        features_koi = [
            'kepid', 'kepoi_name', 'koi_disposition', 'koi_pdisposition',
            'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_period_err2',
            'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact',
            'koi_impact_err1', 'koi_impact_err2', 'koi_duration',
            'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1',
            'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
            'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
            'koi_model_snr', 'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_steff',
            'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1',
            'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra',
            'dec', 'koi_kepmag'
        ]
        self.data = df[features_koi].dropna(subset=['koi_disposition'])

    def preprocess(self):
        # Separate numeric and categorical features
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('koi_disposition') if 'koi_disposition' in numeric_features else None
        categorical_features = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Numeric imputation
        self.num_imputer = SimpleImputer(strategy='median')
        self.data[numeric_features] = self.num_imputer.fit_transform(self.data[numeric_features])

        # Encode categorical features
        for col in categorical_features:
            self.data[col] = self.data[col].fillna('unknown')
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le

        # Encode target
        self.label_encoder_target = LabelEncoder()
        self.y = self.label_encoder_target.fit_transform(self.data['koi_disposition'])
        self.X = self.data.drop(columns=['koi_disposition'])

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
    
    