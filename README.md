# Project

## Prerequisites

Before running this project, you need to:

1. Get the Kaggle API credentials:
   - Create a Kaggle account if you don't have one already
   - Go to your Kaggle account settings (https://www.kaggle.com/account)
   - Scroll down to the API section and click "Create New API Token"
   - This will download a `kaggle.json` file containing your API credentials
   - Place the downloaded `kaggle.json` file in `C:\Users\INTEL\.kaggle`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/aye-shadow/neural-network-acceleration
   cd project
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Download the dataset:
   ```
   python src/extract_dataset.py
   ```
   This will download the MNIST dataset from Kaggle and extract the necessary files to the data directory.

## Usage

For detailed usage instructions, refer to the README files inside each v{x} folder in the src directory:
- src/v1/README.md - Version 1 implementation details

These README files contain version-specific instructions, optimizations, and implementation details.
