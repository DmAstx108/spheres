from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma as special_gamma
from scipy.spatial.distance import euclidean

app = FastAPI(
    title='Sphere'
)

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
    distance = euclidean(center1, center2)
    return distance < radius1 + radius2


# Define the number of samples
num_samples = 100

# Generate random variables for centers (x, y, z) within a specific range
x_centers = np.random.uniform(-100, 100, num_samples)
y_centers = np.random.uniform(-100, 100, num_samples)
z_centers = np.random.uniform(-100, 100, num_samples)

# Generate random angles for symmetric axis orientation in spherical coordinates
phi_angles = np.random.uniform(
    0, 2*np.pi, num_samples)  # Azimuthal angle (phi)
theta_angles = np.random.uniform(0, np.pi, num_samples)  # Polar angle (theta)

# Generate coordinates for 10 spheres with non-overlapping surfaces
all_coordinates = []

for i in range(num_samples):
    center = [x_centers[i], y_centers[i], z_centers[i]]
    radius = np.random.randint(5, 20)  # Random radius between 5 and 20 mm
    x, y, z = generate_sphere_coordinates(center, radius)
    all_coordinates.append((center, radius))

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


@app.get("/pages", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/sphere_image.png")
async def serve_image():
    return FileResponse("sphere_image.png", headers={"Content-Type": "image/png"})
