from flask import Flask, request, render_template, redirect, send_from_directory, abort
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import google.generativeai as genai
from model.k2 import ExoplanetClassifier
from model.toi import TessExoplanetClassifier
from model.koi import KOIClassifier
from sklearn.metrics import confusion_matrix


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


app = Flask(__name__)
UPLOAD_FOLDER = "dataset"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


k2_data = joblib.load("created_model/k2_model.pkl")
koi_data = joblib.load("created_model/koi_model.pkl")
toi_data = joblib.load("created_model/toi_model.pkl")

models = {
    "k2": k2_data["model"],
    "kepler": koi_data["model"],
    "tess": toi_data["model"]
}

imputers = {
    "k2": k2_data["imputer"],
    "kepler": koi_data["imputer"],
    "tess": toi_data["imputer"]
}

features_dict = {
    "k2": k2_data["features"],
    "kepler": koi_data["features"],
    "tess": toi_data["features"]
}

disposition_map = {0: "FALSE POSITIVE", 1: "CONFIRMED", 2: "CANDIDATE"}

def generate_ai_explanation(dataset_type, input_data, predicted_class):
    prompt = f"""
    You are an AI scientist explaining exoplanet predictions in simple language.
    Dataset: {dataset_type}
    Features and values: {input_data}
    Predicted class: {predicted_class}

    Write a 5-6 line explanation that:
    - Uses simple, easy-to-understand language
    - Explains what each important feature means
    - Highlights why the prediction makes sense based on the features
    - Avoid technical jargon or LaTeX
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")
    try:
        response = model.generate_content(prompt)
        explanation = response.text.strip()
        return explanation.replace('$\\text{', '').replace('}$', '')
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def generate_plots(y_true, y_pred, feature_importances=None, feature_names=None):
    class_labels = ["FALSE POSITIVE", "CONFIRMED", "CANDIDATE"]
    n_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    confusion_plot = {
        "type": "bar",
        "labels": class_labels,
        "datasets": [
            {
                "label": f"True {class_labels[i]}",
                "data": cm[i].tolist(),
                "backgroundColor": f"rgba({i*70}, 99, 255, 0.6)"
            } for i in range(n_classes)
        ]
    }

    counts = [sum(y_pred == i) for i in range(n_classes)]
    distribution_plot = {
        "type": "bar",
        "labels": class_labels,
        "datasets": [{
            "label": "Predicted Class Count",
            "data": counts,
            "backgroundColor": ["#ff6384", "#36a2eb", "#ffcd56"]
        }]
    }

    feature_plot = None
    if feature_importances is not None and feature_names is not None:
        feature_plot = {
            "type": "bar",
            "labels": feature_names,
            "datasets": [{
                "label": "Feature Importance",
                "data": feature_importances.tolist(),
                "backgroundColor": "rgba(0, 212, 255, 0.6)"
            }]
        }

    return {
        "confusion_matrix": confusion_plot,
        "class_distribution": distribution_plot,
        "feature_importance": feature_plot
    }


DATASET_FOLDER = "/Users/govindprajapati/Hunting-Exoplanets/dataset"


def generate_dataset_html(files, button_color="#28a745"):
    if not files:
        return "<p>⚠️ No dataset found.</p>"

    html = "<div style='padding:20px; font-family: Arial;'>"
    html += "<h2>📁 Available Datasets</h2>"
    html += "<div style='display: flex; flex-wrap: wrap; gap: 15px;'>"

    for f in files:
        html += f"<a href='/download_dataset/{f}' " \
                f"style='background:{button_color}; color:white; padding:8px 16px; " \
                f"text-decoration:none; border-radius:6px; display:inline-block;'>{f}</a>"

    html += "</div></div>"
    return html

def get_dataset_links():
    if not os.path.exists(DATASET_FOLDER):
        return "<p>❌ Dataset folder not found.</p>"
    
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    return generate_dataset_html(files, button_color="#28a745")  # green buttons



@app.route("/", methods=["GET", "POST"])
def home():
    dataset_links_html = get_dataset_links()
    return render_template("demo.html", dataset_links=dataset_links_html)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("demo.html", dataset_selected=False, dataset_links=get_dataset_links())

    dataset_type = request.form.get("data_set")
    if dataset_type not in models:
        return render_template("demo.html", dataset_selected=False, error="Invalid dataset selected.", dataset_links=get_dataset_links())

    feature_names = features_dict[dataset_type]
    input_data = {}
    for feature in feature_names:
        value = request.form.get(feature)
        if not value:
            return render_template("demo.html", dataset_selected=True, dataset_type=dataset_type,
                                   features=feature_names, error=f"Missing value for {feature}", dataset_links=get_dataset_links())
        try:
            input_data[feature] = float(value)
        except ValueError:
            return render_template("demo.html", dataset_selected=True, dataset_type=dataset_type,
                                   features=feature_names, error=f"Invalid value for {feature}", dataset_links=get_dataset_links())

    df = pd.DataFrame([input_data])
    try:
        X = imputers[dataset_type].transform(df[feature_names])
        prediction = models[dataset_type].predict(X)[0]
        predicted_class = disposition_map.get(prediction, prediction)
        ai_explanation = generate_ai_explanation(dataset_type, input_data, predicted_class)
    except Exception as e:
        return render_template("demo.html", dataset_selected=True, dataset_type=dataset_type,
                               features=feature_names, error=f"Prediction error: {str(e)}", dataset_links=get_dataset_links())

    return render_template("demo.html",
                           dataset_selected=True,
                           dataset_type=dataset_type,
                           features=feature_names,
                           prediction_text=f"Predicted Disposition: {predicted_class}",
                           ai_explanation=ai_explanation,
                           dataset_links=get_dataset_links())

@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("Upload_csv")
    if not file or file.filename == "":
        return render_template("index.html", error="⚠️ No file uploaded.", dataset_links=get_dataset_links())
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(file_path)

    model_name = request.form.get("model")
    data_set = request.form.get("data_set").lower().strip()
    allowed_models = ["randomforest", "xgboost"]

    if model_name not in allowed_models:
        return render_template("index.html", error=f"⚠️ Model {model_name} not supported.", dataset_links=get_dataset_links())

    dataset_alias = {"k2": "k2", "tess": "tess", "kepler": "koi"}
    data_set = dataset_alias.get(data_set)
    if not data_set:
        return render_template("index.html", error="⚠️ Invalid dataset selected.", dataset_links=get_dataset_links())

    n_estimators = int(request.form.get("n_estimators", 100))
    max_depth = int(request.form.get("max_depth", None) or 0)
    min_samples_split = int(request.form.get("min_samples_split", 2))
    min_samples_leaf = int(request.form.get("min_samples_leaf", 1))
    max_features = request.form.get("max_features", None)

    clf_map = {"k2": ExoplanetClassifier, "tess": TessExoplanetClassifier, "koi": KOIClassifier}

    try:
        clf = clf_map[data_set](
            file_path=file_path,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features
        )
        accuracy, report = clf.evaluate()
        plot_divs = generate_plots(
            y_true=clf.y_test,
            y_pred=clf.y_pred,
            feature_importances=getattr(clf.model, "feature_importances_", None),
            feature_names=clf.X.columns.tolist()
        )
        plot_divs = convert_numpy(plot_divs)

        evaluation_text = f"""
        Dataset: {data_set}
        Accuracy: {accuracy}
        Classification Report: {report}
        Confusion Matrix: {plot_divs['confusion_matrix']}
        Class Distribution: {plot_divs['class_distribution']}
        Feature Importance: {plot_divs.get('feature_importance', 'N/A')}
        """
        ai_report = generate_ai_explanation(data_set, {"evaluation": evaluation_text}, "Model Evaluation")

        return render_template(
            "index.html",
            accuracy=accuracy,
            report=report,
            plots=plot_divs,
            ai_report=ai_report,
            dataset_selected=True,
            dataset_type=data_set,
            dataset_links=get_dataset_links()
        )
    except Exception as e:
        return render_template("index.html", error=f"⚠️ Training failed: {str(e)}", dataset_links=get_dataset_links())



@app.route("/download_dataset/<filename>", methods=["GET"])
def download_dataset(filename):
    try:
        return send_from_directory(DATASET_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return f"❌ File '{filename}' not found in {DATASET_FOLDER}"

@app.route("/datasets", methods=["GET"])
def list_datasets():
    if not os.path.exists(DATASET_FOLDER):
        return f"❌ Dataset directory not found: {DATASET_FOLDER}"
    
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    return generate_dataset_html(files, button_color="#007bff")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
