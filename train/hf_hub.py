from huggingface_hub import login


def save_to_hfhub(repo_name, model, processor, hf_token=None):
    if hf_token:
        login(token=hf_token)
    else:
        login() 
        
    processor.push_to_hub(repo_name)
    model.push_to_hub(repo_name)