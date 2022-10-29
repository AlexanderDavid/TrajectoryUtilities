from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="ardi",
    version="0.0.0",
    description="Library and tools for operating on pedestrian trajectories",
    long_description=readme,
    author="Alex Day",
    author_email="alex@alexday.me",
    url='https://github.com/AlexanderDavid/TrajectoryUtilities',
    license=license,
    packages=find_packages(exclude=('tests'))
)