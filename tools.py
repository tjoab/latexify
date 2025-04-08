import yaml
import PIL

# TODO: Make sure yaml paths resolve correctly
def load_config(config_path):
    '''Load YAML config file from file location'''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def open_PIL_image(image_path: str) -> PIL.Image.Image:
    '''
    Open an image from a file path (expected as a JPG or PNG) 
    and return as PIL image object with RGB channels
    '''
    image = PIL.Image.open(image_path)
    
    # Model does not like PNG images due to alpha channel, need to massage in into a JPG-like structure
    if image_path.split('.')[-1].lower() == 'png':
        image = PIL.Image.composite(image, PIL.Image.new('RGB', image.size, 'white'), image)
    return image
