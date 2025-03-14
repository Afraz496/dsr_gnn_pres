import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_ml_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw feature input block
    ax.add_patch(patches.Rectangle((0.1, 0.4), 0.2, 0.2, edgecolor='black', facecolor='lightblue'))
    ax.text(0.2, 0.5, 'Feature Inputs\n(e.g., Weather, Mobility, Demographics)',
            horizontalalignment='center', verticalalignment='center', fontsize=10)

    # Draw arrow to model
    ax.arrow(0.3, 0.5, 0.2, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')

    # Draw model block
    ax.add_patch(patches.Rectangle((0.5, 0.4), 0.2, 0.2, edgecolor='black', facecolor='lightcoral'))
    ax.text(0.6, 0.5, 'Boosted Trees\nModel',
            horizontalalignment='center', verticalalignment='center', fontsize=10)

    # Draw arrow to output
    ax.arrow(0.7, 0.5, 0.2, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')

    # Draw output block
    ax.add_patch(patches.Rectangle((0.9, 0.4), 0.2, 0.2, edgecolor='black', facecolor='lightgreen'))
    ax.text(1.0, 0.5, 'Predictions\n(e.g., Time Series, SHAP)',
            horizontalalignment='center', verticalalignment='center', fontsize=10)

    # Hide axes
    ax.axis('off')
    plt.title('Traditional ML Pipeline for Epidemiological Forecasting', fontsize=14)
    plt.savefig('ml_pipeline_diagram.png')
    plt.close()

draw_ml_pipeline()
