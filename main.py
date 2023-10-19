from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math

from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title='Sphere'
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup
templates = Jinja2Templates(directory="templates")

# Define a function to generate coordinates for a sphere with a given center


def generate_sphere_coordinates(center, radius, resolution=101):
    phi, theta = np.mgrid[0.0:2.0*np.pi:resolution*1j, 0.0:np.pi:resolution*1j]
    x = center[0] + radius*np.sin(theta)*np.cos(phi)
    y = center[1] + radius*np.sin(theta)*np.sin(phi)
    z = center[2] + radius*np.cos(theta)
    return x, y, z

# Function to check if two spheres overlap by their surfaces


def spheres_overlap(center1, radius1, center2, radius2):
    distance = np.linalg.norm(center1 - center2)
    return distance < radius1 + radius2

# Generate random coordinates using Gaussian distribution


def generate_random_coordinates_gauss(mean, std_dev, num_samples):
    return np.random.normal(mean, std_dev, num_samples)


def generate_sphere_parameters(num_spheres):
    smallest_radius = np.random.randint(5, 20)
    largest_radius = np.random.randint(smallest_radius, 20)

    smallest_center = np.random.uniform(-100, 100, size=3)
    largest_center = np.random.uniform(-100, 100, size=3)

    return (smallest_center, smallest_radius), (largest_center, largest_radius)


# Generate parameters for smallest and largest spheres
(smallest_center, smallest_radius), (largest_center,
                                     largest_radius) = generate_sphere_parameters(100)

# Generate coordinates for 10 spheres with non-overlapping surfaces
all_coordinates = []

for _ in range(10):
    while True:
        # Random radius between smallest and largest mm
        radius = np.random.randint(smallest_radius, largest_radius)

        # Generate random coordinates for the center using Gaussian distribution
        center_x, center_y, center_z = generate_random_coordinates_gauss(
            0, 10, 3)
        center = np.array([center_x, center_y, center_z])

        overlap = False

        for other_center, other_radius in all_coordinates:
            if spheres_overlap(center, radius, other_center, other_radius):
                overlap = True
                break

        if not overlap:
            break

    x, y, z = generate_sphere_coordinates(center, radius)
    all_coordinates.append((center, radius))

# Sort spheres by their radii from smallest to largest
all_coordinates = sorted(all_coordinates, key=lambda x: x[1])

# Generate and save the image
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

for center, radius in all_coordinates:
    x, y, z = generate_sphere_coordinates(center, radius)
    ax.scatter(x, y, z, s=1, c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'3D Plot of 10 Spheres with Non-Overlapping Surfaces')
plt.savefig("sphere_image.png")
plt.close()

coordinates = [coord for coord, _ in all_coordinates]
dimensions = [radius for _, radius in all_coordinates]

# Create a DataFrame for coordinates
df_coordinates = pd.DataFrame(coordinates, columns=['X', 'Y', 'Z'])

# Create a DataFrame for dimensions
df_dimensions = pd.DataFrame(dimensions, columns=['Radius'])

# Assuming df_coordinates and df_dimensions are DataFrames containing the coordinates and dimensions

df_coordinates.to_excel('coordinates.xlsx', index=False)
df_dimensions.to_excel('dimensions.xlsx', index=False)

# Combine all the information into one DataFrame
df_combined = df_coordinates.copy()
df_combined['Radius'] = df_dimensions['Radius']
df_combined['Diameter'] = 2 * df_combined['Radius']  # Calculate diameter

# Calculate the dimensions of the restricted space containing all spheres
min_x = df_combined['X'].min() - df_combined['Radius'].max()
max_x = df_combined['X'].max() + df_combined['Radius'].max()
min_y = df_combined['Y'].min() - df_combined['Radius'].max()
max_y = df_combined['Y'].max() + df_combined['Radius'].max()
min_z = df_combined['Z'].min() - df_combined['Radius'].max()
max_z = df_combined['Z'].max() + df_combined['Radius'].max()

restricted_dimensions = {
    'Min X': min_x,
    'Max X': max_x,
    'Min Y': min_y,
    'Max Y': max_y,
    'Min Z': min_z,
    'Max Z': max_z
}


# Calculate the volume of each sphere
volumes = [(4/3) * math.pi * radius**3 for _, radius in all_coordinates]

# Calculate the total volume
total_volume = sum(volumes)

volume_restricted_space = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

# Create a DataFrame for the volume of restricted space
df_restricted_volume = pd.DataFrame(
    {'Attribute': ['Volume of Restricted Space'], 'Value': [volume_restricted_space]})

# Concatenate the restricted volume DataFrame with the dimensions DataFrame
df_dimensions = pd.concat(
    [df_dimensions, df_restricted_volume], ignore_index=True)

# Save dimensions DataFrame to 'dimensions.xlsx'
df_dimensions.to_excel('dimensions.xlsx', index=False)


# Add restricted dimensions to the DataFrame
df_combined = df_combined._append(restricted_dimensions, ignore_index=True)

# Save the combined DataFrame to a new Excel file
df_combined.to_excel('combined_data.xlsx', index=False)

# volume_restricted_space = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

# # Add the volume of the restricted space to the dimensions DataFrame
# df_dimensions.loc[len(df_dimensions)] = [
#     'Volume of Restricted Space', volume_restricted_space]


@app.get("/pages", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("sphere_visualization.html", {"request": request})


@app.get("/sphere_image.png")
async def serve_image():
    return FileResponse("sphere_image.png", headers={"Content-Type": "image/png"})
