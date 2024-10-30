"""Install the llm_merging library"""

from setuptools import setup

setup(
    name="llm_merging",
    version="1.0",
    description="Starter code for llm_merging",
    install_requires=[
        "torch",
        "ipdb",
        "transformers",  # Required for model handling
        "numpy",         # For numerical operations
        "scipy",         # For advanced mathematical functions
        "datasets",      # For dataset handling
        "tqdm",          # For progress tracking
        "pandas",        # For data manipulation and CSV handling
        "safetensors"    # For saving merged models in the Safetensors format
    ],
    packages=["llm_merging"],
    entry_points={
        "llm_merging.merging.Merges": [
            "flant5_geomed = llm_merging.merging.FlanT5GeoMed:FlanT5GeoMed"
        ]
    },
)
