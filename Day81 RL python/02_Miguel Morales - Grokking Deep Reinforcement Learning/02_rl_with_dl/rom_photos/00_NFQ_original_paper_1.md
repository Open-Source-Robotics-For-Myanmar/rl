```mermaid
graph TD
    subgraph Phase1 [Data Gathering]
        Env[Environment] -->|State and Reward| Agent[Agent]
        Agent -->|Action| Env
        Agent -->|Collect Transition| Buffer[(Replay Buffer D)]
    end

    subgraph Phase2 [Mini Batch Sampling]
        Buffer -->|Sample Batch| Batch[Transition Sample]
    end

    subgraph Phase3 [Q Network and Target]
        Batch -->|State and Action| MLP[Current Q Network]
        MLP --> Q_pred[Predicted Q Value]

        Batch -->|Next State| Target_Net[Target Q Network]
        Target_Net --> Max_Q[Max Q Value]
        Batch -->|Reward| Target_Y[Target Value Y]
        Max_Q --> Target_Y
    end

    subgraph Phase4 [Optimization]
        Q_pred --> Loss[MSE Loss]
        Target_Y --> Loss
        Loss --> Opt[Gradient Descent Update]
        Opt -->|Update Weights| MLP
    end