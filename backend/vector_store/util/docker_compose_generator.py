import yaml
from pathlib import Path
from typing import List

# Custom YAML representer to force double quotes for strings
def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

# Add the custom representer to the YAML dumper
yaml.add_representer(str, quoted_presenter)

def generate_docker_compose(dimensions: List[int], base_port: int = 5080) -> str:
    """
    Generate docker-compose.yaml content for Pinecone indexes.
    
    Args:
        dimensions: List of dimensions for different indexes
        base_port: Starting port number
    """
    services = {}
    
    for i, dim in enumerate(dimensions):
        port = base_port + i
        service_name = f"index{i+1}"
        
        services[service_name] = {
            "image": "ghcr.io/pinecone-io/pinecone-index:latest",
            "environment": {
                "PORT": port,
                "INDEX_TYPE": "serverless",
                "DIMENSION": dim,
                "METRIC": "euclidean"
            },
            "ports": [f"{port}:{port}"],  # No explicit wrapping in quotes here
            "platform": "linux/amd64"
        }
    
    docker_compose = {
        "version": "3",
        "services": services
    }
    
    return yaml.dump(docker_compose, default_flow_style=False)

def write_docker_compose(dimensions: List[int], output_path: Path = None) -> Path:
    """
    Generate and write docker-compose.yaml file.
    
    Args:
        dimensions: List of dimensions for different indexes
        output_path: Path to write the docker-compose.yaml file
    
    Returns:
        Path to the generated docker-compose.yaml file
    """
    if output_path is None:
        # Go up one more directory level to reach project root
        output_path = Path(__file__).parent.parent.parent.parent / "docker-compose.yaml"
    
    content = generate_docker_compose(dimensions)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    return output_path