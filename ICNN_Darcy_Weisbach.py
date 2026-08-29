# -*- coding: utf-8 -*-
"""ICNN-based modelling of friction loss in water distribution networks.

This script generates synthetic pipe-flow scenarios from the Darcy-Weisbach
relationship and trains an Input-Convex Neural Network (ICNN) to approximate
friction loss from flow rate and pipe diameter.

Developed as part of Yitao Shi's Master's thesis at Wageningen University &
Research.
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# -----------------------------------------------------------------------------
# Physical model
# -----------------------------------------------------------------------------

def darcy_weisbach_friction_loss(flow_rate, diameter, length, roughness):
    """Calculate friction loss and identify the flow condition.

    Parameters
    ----------
    flow_rate : float
        Flow rate in m^3/s.
    diameter : float
        Pipe diameter in m.
    length : float
        Pipe length in m.
    roughness : float
        Pipe roughness height in m.

    Returns
    -------
    friction_loss : float
        Friction loss in m.
    flow_condition : str
        Flow regime classified as laminar, transitional, or turbulent.
    """
    gravity = 9.81
    viscosity = 1.002e-3
    density = 998.2

    # Reynolds number
    reynolds_number = (4 * flow_rate * density) / (
        math.pi * diameter * viscosity
    )

    if reynolds_number < 2000:
        flow_condition = "laminar"
    elif reynolds_number < 4000:
        flow_condition = "transitional"
    else:
        flow_condition = "turbulent"

    # Friction factor
    if flow_condition == "laminar":
        friction_factor = 64 / reynolds_number
    elif flow_condition == "transitional":
        relative_roughness = roughness / diameter
        friction_factor = (
            1
            / (
                -1.8
                * math.log10(
                    relative_roughness / 3.7
                    + 5.74 / (reynolds_number**0.9)
                )
            )
        ) ** 2
    else:
        # Fixed-point iteration for the Colebrook-White equation.
        friction_factor = 0.02
        tolerance = 1e-6

        while True:
            new_friction_factor = 1 / (
                -2
                * math.log10(
                    (2.51 / (reynolds_number * math.sqrt(friction_factor / 2)))
                    + (roughness / (3.71 * diameter))
                )
            ) ** 2

            if abs(new_friction_factor - friction_factor) < tolerance:
                break
            friction_factor = new_friction_factor

        friction_factor = new_friction_factor

    # Friction loss. The factor 1000 is retained from the original thesis code.
    friction_loss = (
        8 * length * friction_factor * flow_rate**2
    ) / (math.pi**2 * gravity * diameter**5) * 1000

    return friction_loss, flow_condition


# -----------------------------------------------------------------------------
# ICNN model
# -----------------------------------------------------------------------------

def forward(x, weights, biases, Vs):
    """Forward pass through the ICNN hidden layers."""
    z = torch.relu(torch.matmul(x, weights[0]) + biases[0])

    for i in range(1, len(weights)):
        z = torch.relu(
            torch.matmul(z, weights[i])
            + torch.matmul(x, Vs[i])
            + biases[i]
        )

    return z


def predict(x, weights, biases, Vs, output_weights, output_bias, output_Vs):
    """Generate ICNN predictions without tracking gradients."""
    with torch.no_grad():
        output = forward(x, weights, biases, Vs)
        output = (
            torch.matmul(output, output_weights)
            + torch.matmul(x, output_Vs)
            + output_bias
        )
    return output


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------

def generate_dataset(n_scenarios=100_000):
    """Generate synthetic pipe-flow scenarios and reference friction losses."""
    length = 1.0
    roughness = 0.0002

    scenarios = []
    for _ in range(n_scenarios):
        flow_rate = random.uniform(0, 0.589)
        diameter = round(random.uniform(0.15, 0.5) * 50) / 50
        scenarios.append((flow_rate, diameter))

    results = []
    for flow_rate, diameter in scenarios:
        friction_loss, flow_condition = darcy_weisbach_friction_loss(
            flow_rate, diameter, length, roughness
        )
        results.append((flow_rate, diameter, friction_loss, flow_condition))

    return pd.DataFrame(
        results,
        columns=[
            "Flow Rate (m^3/s)",
            "Diameter (m)",
            "Friction Loss (m)",
            "Flow Condition",
        ],
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def initialise_icnn():
    """Initialise ICNN weights, biases, and skip-connection parameters."""
    input_size = 2
    hidden_layers_sizes = [5, 10, 5, 2]
    output_size = 1

    weights = [
        torch.randn(
            input_size,
            hidden_layers_sizes[0],
            dtype=torch.float32,
            requires_grad=True,
        )
    ]
    biases = [
        torch.randn(
            hidden_layers_sizes[0], dtype=torch.float32, requires_grad=True
        )
    ]
    Vs = [
        torch.randn(
            input_size,
            hidden_layers_sizes[0],
            dtype=torch.float32,
            requires_grad=True,
        )
    ]

    for i in range(1, len(hidden_layers_sizes)):
        weights.append(
            torch.randn(
                hidden_layers_sizes[i - 1],
                hidden_layers_sizes[i],
                dtype=torch.float32,
                requires_grad=True,
            )
        )
        biases.append(
            torch.randn(
                hidden_layers_sizes[i],
                dtype=torch.float32,
                requires_grad=True,
            )
        )
        Vs.append(
            torch.randn(
                input_size,
                hidden_layers_sizes[i],
                dtype=torch.float32,
                requires_grad=True,
            )
        )

    output_weights = torch.randn(
        hidden_layers_sizes[-1], output_size, dtype=torch.float32, requires_grad=True
    )
    output_bias = torch.randn(
        output_size, dtype=torch.float32, requires_grad=True
    )
    output_Vs = torch.randn(
        input_size, output_size, dtype=torch.float32, requires_grad=True
    )

    return weights, biases, Vs, output_weights, output_bias, output_Vs


def train_icnn(x_train, y_train, x_val, y_val, epochs=10_000, learning_rate=0.002):
    """Train the ICNN and return the trained parameters and loss history."""
    weights, biases, Vs, output_weights, output_bias, output_Vs = initialise_icnn()

    optimizer = torch.optim.Adam(
        [
            {"params": weights + biases + Vs},
            {"params": [output_weights, output_bias, output_Vs]},
        ],
        lr=learning_rate,
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Project selected weights to non-negative values to maintain the ICNN
        # convexity constraint.
        for weight in weights[1:]:
            weight.data.clamp_(0)
        output_weights.data.clamp_(0)

        # Training pass
        output_train = forward(x_train, weights, biases, Vs)
        output_train = (
            torch.matmul(output_train, output_weights)
            + torch.matmul(x_train, output_Vs)
            + output_bias
        )
        train_loss = criterion(output_train, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Re-project constrained weights after the update.
        for weight in weights[1:]:
            weight.data.clamp_(0)
        output_weights.data.clamp_(0)

        # Validation pass
        with torch.no_grad():
            output_val = forward(x_val, weights, biases, Vs)
            output_val = (
                torch.matmul(output_val, output_weights)
                + torch.matmul(x_val, output_Vs)
                + output_bias
            )
            val_loss = criterion(output_val, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"Training Loss = {train_loss.item()}, "
                f"Validation Loss = {val_loss.item()}"
            )

    return (
        weights,
        biases,
        Vs,
        output_weights,
        output_bias,
        output_Vs,
        train_losses,
        val_losses,
    )


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main():
    # Generate reference data.
    df = generate_dataset()

    # Prepare model inputs and outputs.
    input_data = df[["Flow Rate (m^3/s)", "Diameter (m)"]].to_numpy()
    output_data = df[["Friction Loss (m)"]].to_numpy()

    # Split data into training, validation, and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, output_data, test_size=0.01, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42
    )

    # Normalize using the training data only.
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()
    x_train = scaler_input.fit_transform(x_train)
    x_val = scaler_input.transform(x_val)
    x_test = scaler_input.transform(x_test)
    y_train = scaler_output.fit_transform(y_train)
    y_val = scaler_output.transform(y_val)
    y_test = scaler_output.transform(y_test)

    # Convert to tensors.
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Train ICNN.
    (
        weights,
        biases,
        Vs,
        output_weights,
        output_bias,
        output_Vs,
        train_losses,
        val_losses,
    ) = train_icnn(x_train, y_train, x_val, y_val)

    # Evaluate on the held-out test set only after training.
    output_test = predict(
        x_test,
        weights,
        biases,
        Vs,
        output_weights,
        output_bias,
        output_Vs,
    )

    predicted = scaler_output.inverse_transform(output_test.numpy())
    y_test_original = scaler_output.inverse_transform(y_test.numpy())

    differences = predicted.flatten() - y_test_original.flatten()
    abs_diff = np.abs(differences)
    mean_diff = np.mean(abs_diff)

    # Plot model output vs. reference data for flow rate.
    flow_rate = x_test[:, 0].numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(flow_rate, predicted, label="Model Output")
    plt.scatter(flow_rate, y_test_original, label="Reference Data")
    plt.xlabel("Flow rate (normalized)")
    plt.ylabel("Friction loss")
    plt.title("Model Output vs. Reference Data (Flow Rate)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot absolute error for flow rate.
    threshold = np.percentile(abs_diff, 90)
    highlight_idx = abs_diff >= threshold

    plt.figure(figsize=(8, 6))
    plt.scatter(flow_rate, abs_diff, label="Absolute Error")
    plt.axhline(mean_diff, linestyle="-", label="Mean Absolute Error", linewidth=2)
    plt.scatter(
        flow_rate[highlight_idx],
        abs_diff[highlight_idx],
        label="Top 10% Error",
    )
    plt.xlabel("Flow rate (normalized)")
    plt.ylabel("Absolute friction-loss error")
    plt.title("Friction-Loss Error (Flow Rate)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot model output vs. reference data for diameter.
    diameter = x_test[:, 1].numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(diameter, predicted, label="Model Output", alpha=0.6)
    plt.scatter(diameter, y_test_original, label="Reference Data", alpha=0.6)
    plt.xlabel("Diameter (normalized)")
    plt.ylabel("Friction loss")
    plt.title("Model Output vs. Reference Data (Diameter)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot absolute error for diameter.
    plt.figure(figsize=(8, 6))
    plt.scatter(diameter, abs_diff, label="Absolute Error")
    plt.axhline(mean_diff, linestyle="-", label="Mean Absolute Error", linewidth=2)
    plt.scatter(
        diameter[highlight_idx],
        abs_diff[highlight_idx],
        label="Top 10% Error",
    )
    plt.xlabel("Diameter (normalized)")
    plt.ylabel("Absolute friction-loss error")
    plt.title("Friction-Loss Error (Diameter)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training and validation loss.
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save trained parameters.
    torch.save(
        {
            "weights": weights,
            "biases": biases,
            "Vs": Vs,
            "output_weights": output_weights,
            "output_bias": output_bias,
            "output_Vs": output_Vs,
        },
        "trained_model.pth",
    )

    print(f"Mean absolute error on the held-out test set: {mean_diff}")


if __name__ == "__main__":
    main()
