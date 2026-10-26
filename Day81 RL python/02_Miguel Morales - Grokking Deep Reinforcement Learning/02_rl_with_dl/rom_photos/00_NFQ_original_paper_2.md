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

    subgraph Phase3 [Q Networks and Target Calculation]
        Batch -->|State s and Action a| Q_Current[Current Q-Network<br/>Updates via Gradient Descent]
        Q_Current --> Q_pred[Predicted Q Value]

        Batch -->|Next State s'| Q_Target[Target Generation<br/>Fixed weights from previous iteration]
        Q_Target --> Max_Q[Max Q Value]
        Batch -->|Reward r| Target_Y[Target Value Y]
        Max_Q --> Target_Y
    end

    subgraph Phase4 [Optimization]
        Q_pred --> Loss[MSE Loss]
        Target_Y --> Loss
        Loss --> Opt[Gradient Descent Update]
        Opt -->|Updates Weights| Q_Current
    end