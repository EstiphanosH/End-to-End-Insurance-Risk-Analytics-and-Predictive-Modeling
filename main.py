# Main application entry point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """Main application function"""
    print("Data Analysis Project")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Sample data visualization
    data = pd.Series(np.random.randn(1000))
    data.plot(kind='hist', title='Sample Distribution')
    plt.savefig('reports/figures/sample_distribution.png')
    print("Generated sample visualization")

if __name__ == "__main__":
    main()
