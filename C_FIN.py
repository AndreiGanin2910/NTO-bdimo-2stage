import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings
import os
import zipfile

zip_path = "D_data/NTO_BDML_2C_classification.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")


warnings.filterwarnings('ignore')

train_df = pd.read_csv("D_data/train.csv")
test_df = pd.read_csv("D_data/test.csv")
sample_submission = pd.read_csv("D_data/sample_submition.csv")


train_len = len(train_df)
test_len = len(test_df)
y = train_df['target']
train_df = train_df.drop('target', axis=1)


data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)


cols_to_drop = ['f5', 'f15', 'f37']
data = data.drop(cols_to_drop, axis=1)


categorical_features = data.select_dtypes(include=['object']).columns


for col in categorical_features:
    data[col] = data[col].fillna('miss')


for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])


target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)


X_train = data[:train_len]
X_test = data[train_len:]


lgb_clf = lgb.LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    n_estimators=4000,
    learning_rate=0.01,
    num_leaves=45,
    max_depth=-1,
    min_child_samples=60,
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.5,
    reg_lambda=5,
    class_weight='balanced'
)

print("Обучение LGBMClassifier")
lgb_clf.fit(X_train, y_encoded,
            categorical_feature=[col for col in X_train.columns if col in categorical_features])
print("Обучение завершено.")


print("Предсказание")
predictions_encoded = lgb_clf.predict(X_test)

predictions = target_encoder.inverse_transform(predictions_encoded)
submission = pd.DataFrame({'target': predictions})
submission.to_csv('D_outputs/submission.csv', index=False)
print("Готово")