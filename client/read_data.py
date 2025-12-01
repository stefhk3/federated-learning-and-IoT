from torch.utils.data import Subset
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import IotDataset

def read_data(data_path):
    df = pd.read_csv(data_path, delimiter='|')
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
    df2=df["label",features]
    dataset = IotDataset(data_path)
    return dataset