# A Sliding Window Approach to Designing Self-Adjusting Networks

This repository includes the code referenced in the bachelor thesis titled **"A Sliding Window Approach to Designing Self-Adjusting Networks"**.

## Overview

This repository contains the implementation for my bachelor thesis. The goal of this project is to explore the use of a sliding window approach in creating networks that can self-adjust to optimize performance based on dynamic conditions.

## Table of Contents

- [Introduction](#introduction)
- [Components](#components)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)


## Introduction

In this thesis, we present a novel approach to network design using a sliding window technique. The primary aim is to create networks that can dynamically adjust themselves to changing conditions, thereby improving performance and reliability. This approach has been applied to various scenarios, and the results are documented in the accompanying thesis document.

## Components

- **Self-Adjusting Network (SAN) Algorithms**: These algorithms automatically adjust the network topology based on communication demand. The algorithms include SplayNet, Lazy SplayNet, Variable Overlapping Sliding Window SplayNet (VOWS), and its no-reset version NR-VOWS. These are implemented in the file `algorithms.py` located in the `algorithms` folder.
- **Network Topologies**: The network topology utilized in this thesis is the BSt network. Variants include balanced, frequency-based, randomly shuffled, and static optimal, as described in the thesis. The structure is detailed in the `Node.py` and `SplayNetwork.py` files within the `topology` folder.
- **Communication Sequences**: These sequences are used to run the SANs and are saved as .csv files. The `csv` folder contains randomly created traces with varying temporal localities for network sizes ranging from 100 to 1000 in steps of 100. Real-world data traces are stored in the `data` folder. The `spatial_data_10_5` and `spatial_10_6` folders contain communication sequences of sizes 10^5 and 10^6 respectively, with varying spatial localities. Similarly, the `temporal_data_10_5` and `temporal_10_6` folders contain sequences with varying temporal localities.
- **Communication Requests**: These requests, defined in the `CommunicationRequest.py` file in the `topology` folder, are the elements within the communication sequences.
- **Performance Evaluation**: Various hyperparameters can be set for the simulation, which is executed through the `main.py` file. The results are stored in the `output` folder, with the evaluated results saved in the `evaluation_data` folder. The filenames include the topology type, distribution type, and network size.
- **Scalability**: The approach is applicable to different types and sizes of networks.

## Installation

### Prerequisites

- Python 3.11
- Required libraries (listed in `requirements.txt`)

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/tayosgit/sliding-window-thesis
    cd sliding-window-thesis
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Network Simulation

To run the network simulation, go into the `main.py` file and set the hyperparameters as desired. Then, execute the following command:

```bash
python main.py
```

## Contact

For any questions or inquiries, please contact me at:

- **Name**: Maria Cole
- **Email**: m.cole@campus.tu-berlin.de
- **LinkedIn**: [Your LinkedIn Profile](www.linkedin.com/in/maria-cole-734857234)
