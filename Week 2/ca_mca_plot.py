import numpy as np
import matplotlib.pyplot as plt
import prince
import pandas as pd

def mca_plot(mca, df, select=None, title="", plot="cols"):
    """
    Plot MCA results with flexible display options.

    Parameters
    ----------
    mca : prince.MCA
        The fitted MCA object.
    df : pd.DataFrame
        The dataframe used to compute the MCA.
    select : list or None
        Optional list of columns used in the MCA (if a subset of df was used).
    title : str, optional
        Title of the plot.
    plot : {'rows', 'cols', 'both'}, optional
        What to plot. Default is 'cols' (columns only).
    """
    # Subset if provided
    data = df[select] if select is not None else df

    # Coordinates
    row_coords = mca.row_coordinates(data)
    col_coords = mca.column_coordinates(data)

# Explained inertia (%)
    eigvals = np.array(mca.cumulative_percentage_of_variance_)
    eig1 = eigvals[0]
    eig2 = eigvals[1] - eigvals[0]
    eigvals = np.array([eig1, eig2])
    dim1 = eigvals[0] if len(eigvals) > 0 else 0
    dim2 = eigvals[1] if len(eigvals) > 1 else 0



    # Plot setup
    plt.figure(figsize=(8, 8))

    # Plot according to the 'plot' argument
    if plot in ("rows", "both"):
        plt.scatter(row_coords[0], row_coords[1], label="Rows", color="steelblue", alpha=0.5)
        for i, row_name in enumerate(row_coords.index):
            plt.text(
                row_coords.iloc[i, 0] + 0.02,
                row_coords.iloc[i, 1] + 0.02,
                str(row_name),
                fontsize=8,
                color="steelblue"
            )

    if plot in ("cols", "both"):
        plt.scatter(col_coords[0], col_coords[1], label="Columns", color="crimson", marker="x", s=80)
        for i, col_name in enumerate(col_coords.index):
            plt.text(
                col_coords.iloc[i, 0] + 0.02,
                col_coords.iloc[i, 1] + 0.02,
                str(col_name),
                fontsize=9,
                color="crimson"
            )

    # Style
    plt.legend()
    plt.xlabel(f"Dimension 1 ({dim1:.1f}% inertia)")
    plt.ylabel(f"Dimension 2 ({dim2:.1f}% inertia)")
    plt.title(f"MCA Plot: {title}")
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def ca_plot(ca, df, title="", plot = "both"):
    # Plot the results
    row_coords = ca.row_coordinates(df)
    col_coords = ca.column_coordinates(df)

    # Explained inertia (%)
    eigvals = np.array(ca.eigenvalues_)
    inertia = 100 * eigvals / eigvals.sum()
    dim1 = inertia[0] if len(inertia) > 0 else 0
    dim2 = inertia[1] if len(inertia) > 1 else 0

       # Plot setup
    plt.figure(figsize=(8, 8))

    # Plot according to the 'plot' argument
    if plot in ("rows", "both"):
        plt.scatter(row_coords[0], row_coords[1], label="Rows", color="steelblue", alpha=0.5)
        for i, row_name in enumerate(row_coords.index):
            plt.text(
                row_coords.iloc[i, 0] + 0.02,
                row_coords.iloc[i, 1] + 0.02,
                str(row_name),
                fontsize=8,
                color="steelblue"
            )

    if plot in ("cols", "both"):
        plt.scatter(col_coords[0], col_coords[1], label="Columns", color="crimson", marker="x", s=80)
        for i, col_name in enumerate(col_coords.index):
            plt.text(
                col_coords.iloc[i, 0] + 0.02,
                col_coords.iloc[i, 1] + 0.02,
                str(col_name),
                fontsize=9,
                color="crimson"
            )
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    # Axis labels include explained variance (%)
    plt.xlabel(f"Dimension 1 ({dim1:.1f}% inertia)")
    plt.ylabel(f"Dimension 2 ({dim2:.1f}% inertia)")
    plt.title("Correspondence Analysis: "+title, fontsize=12)
    plt.gca().set_aspect("equal", adjustable="datalim")
