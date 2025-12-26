In our simulation, we create independent client objects, each representing a real IoT device. Each client:

1.
Receives a data partition (simulating local data on the device)
2.
Has its own model copy (initialized with same weights as global model)
3.
Performs local training on its own data for N epochs
4.
Sends only model weights to the server (not raw data)
5.
Receives aggregated weights from server after each round

The server:

Aggregates weights using FedAvg (weighted average based on data size)
Broadcasts updated weights to all clients
Coordinates communication rounds

This accurately simulates federated learning paradigm:

Data stays local (privacy preserved)
Only model updates are shared
Each client trains independently on their own data partition
Global model improves through collaborative learning"
Key Point: The simulation uses Python objects to represent what would be separate devices in real FL. This is standard practice for FL research before deployment on actual devices.
