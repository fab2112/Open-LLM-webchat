# Open-LLM-webchat - Process model selector dropdowns
import settings
import gradio as gr

from collections import defaultdict

def get_owners_to_models() -> defaultdict:
    """
    Map model providers to their respective model identifiers.

    This function iterates over the list of models defined in `settings.MODELS`,
    extracts the provider (the part before the first underscore) and the model
    identifier (the part after the first underscore, and groups the models by their providers.

    Returns:
        defaultdict[list]: A dictionary mapping each provider to a list of their model identifiers.
    """
    # Create dictionary to map owners to models
    owner_to_models = defaultdict(list)
    for model in settings.MODELS:
        # Extract the owner (before the first "_")
        owner = model.split("_")[0]
        # Extract the model name (after the "_")  
        model_name = model[len(owner) + 1:]  
        owner_to_models[owner].append(model_name)
    return owner_to_models

def update_model_name_dropdown(selected_owner: str) -> gr.Dropdown:
    """
    Update the model dropdown options based on the selected provider.

    This function retrieves the mapping of providers to their models,
    selects the models corresponding to the chosen provider, and
    returns a Gradio Dropdown component with these options.

    Args:
        selected_owner (str): The provider selected by the user.

    Returns:
        gr.Dropdown: A Gradio Dropdown component populated with the models of the selected provider.
    """
    owner_to_models = get_owners_to_models()
    models = owner_to_models[selected_owner]
    
    return gr.Dropdown(choices=models, value=models[0])  

def get_full_model_name(owner: str, model_name: str) -> str:
    """
    Combine the provider and model name into a full model identifier.

    This function concatenates the provider (owner) and the model name
    with an underscore to produce the complete model identifier.

    Args:
        owner (str): The model provider or owner.
        model_name (str): The name of the model.

    Returns:
        str: The full model identifier in the format "owner_modelname".
    """
    return f"{owner}_{model_name}"