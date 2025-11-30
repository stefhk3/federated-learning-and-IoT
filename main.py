import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ipaddress
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    log_loss, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
import psutil
from sklearn.linear_model import Perceptron

file_path = 'CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv'
df = pd.read_csv(file_path, delimiter='|')
df.rename(columns={
    'ts': 'timestamp',
    'uid': 'unique_id',
    'id.orig_h': 'origin_host_ip',
    'id.orig_p': 'origin_host_port',
    'id.resp_h': 'response_host_ip',
    'id.resp_p': 'response_host_port',
    'proto': 'protocol',
    'orig_bytes': 'origin_bytes',
    'resp_bytes': 'response_bytes',
    'conn_state': 'connection_state',
    'local_orig': 'is_local_origin',
    'local_resp': 'is_local_response',
    'orig_pkts': 'origin_packet_count',
    'orig_ip_bytes': 'origin_ip_bytes',
    'resp_pkts': 'response_packet_count',
    'resp_ip_bytes': 'response_ip_bytes',
}, inplace=True)
numeric_cols = ['duration', 'origin_bytes', 'response_bytes', 'missed_bytes',
                'origin_packet_count', 'origin_ip_bytes', 'response_packet_count', 'response_ip_bytes']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

print(df.isnull().sum())

df[['origin_bytes', 'response_bytes']] = df[['origin_bytes', 'response_bytes']].fillna(0).astype(int)
df[['service', 'history']] = df[['service', 'history']].fillna('unknown')
df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df.iloc[1:].reset_index(drop=True)
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df.drop('timestamp', axis=1, inplace=True)
df.replace('-', np.nan, inplace=True)
df = df[df['label'].isin(['Malicious', 'Benign'])]
df['label'] = df['label'].map({'Malicious': 1, 'Benign': 0})
df['origin_host_ip'] = df['origin_host_ip'].apply(lambda ip: int(ipaddress.IPv4Address(ip)) if pd.notnull(ip) else 0)
df['response_host_ip'] = df['response_host_ip'].apply(lambda ip: int(ipaddress.IPv4Address(ip)) if pd.notnull(ip) else 0)

features = [
    'origin_host_port',
    'response_host_port',
    'origin_ip_bytes',
    'response_ip_bytes',
    'duration',
    'origin_bytes',
    'response_bytes',
    'origin_packet_count',
    'response_packet_count',
      'response_host_ip',
    'origin_host_ip',
]
X = df[features]
y = df["label"]
X['duration'].fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results_df = pd.DataFrame(columns=[
    "Model", "Accuracy", "F1 Score", "Precision", "Recall", "AUC-ROC",
    "Log Loss", "MCC", "Balanced Accuracy", "Training Time (s)", "Memory (MB)"
])
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

""" model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
start_memory = psutil.Process().memory_info().rss / 1024 ** 2  # Memory in MB

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1024, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_proba = model.predict(X_test)

f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
log_loss_value = log_loss(y_test, y_proba) if y_proba is not None else None

train_time = time.time() - start_time
end_memory = psutil.Process().memory_info().rss / 1024 ** 2
memory_usage = end_memory - start_memory

results_df = pd.DataFrame([{
    "Model": "Sequential",
    "Accuracy": accuracy,
    "F1 Score": f1,
    "Precision": precision,
    "Recall": recall,
    "AUC-ROC": auc_roc,
    "Log Loss": log_loss_value,
    "MCC": mcc,
    "Balanced Accuracy": balanced_accuracy,
    "Training Time (s)": train_time,
    "Memory (MB)": memory_usage
}])

print("Sequential model evaluated.")
print(results_df)
 """
model = Perceptron()

start_time = time.time()
start_memory = psutil.Process().memory_info().rss / 1024 ** 2  # Memory in MB

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

# Core metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# Probability-based metrics
auc_roc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None else None
log_loss_value = log_loss(y_test, y_proba) if y_proba is not None else None

# Resource usage
train_time = time.time() - start_time
end_memory = psutil.Process().memory_info().rss / 1024 ** 2
memory_usage = end_memory - start_memory

# Append results to DataFrame
results_df = pd.concat([results_df, pd.DataFrame([{
    "Model": Perceptron,
    "Accuracy": accuracy,
    "F1 Score": f1,
    "Precision": precision,
    "Recall": recall,
    "AUC-ROC": auc_roc,
    "Log Loss": log_loss_value,
    "MCC": mcc,
    "Balanced Accuracy": balanced_accuracy,
    "Training Time (s)": train_time,
    "Memory (MB)": memory_usage
}])], ignore_index=True)

print("Perceptron evaluated.")
print(results_df)