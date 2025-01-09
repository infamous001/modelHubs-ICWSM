from tqdm import tqdm
from huggingface_hub import HfApi, ModelFilter
import json
import requests
from multiprocessing import Pool
from itertools import cycle

hf_api = HfApi(
    endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
    token="",  # Token is not persisted on the machine.
)

# filt = ModelFilter(task="text-classification")

# Fetch model lists for NLI
# models_list_nli = hf_api.list_models(
#     filter="text-classification", full=True, cardData=True, fetch_config=True, sort="downloads", direction=-1, tags='natural-language-inference'
# )
# Fetch model info of all NLI models with "nli" keyword in their names or descriptions
models_list_nli_kw = hf_api.list_models(
    filter='text-classification', search="nli", full=True, cardData=True, fetch_config=True, sort="downloads",direction=-1
)

# Extract model IDs
# nli_ids = [nli.modelId for nli in models_list_nli]
nli_kw = [nli.modelId for nli in models_list_nli_kw]

# nli_non_keyword = set(nli_ids).difference(nli_kw)

# nli_non_keyword_models = list(nli_non_keyword)
nli_keyword_models = nli_kw

def howManyLabelModels(token, mname):
    API_URL = f"https://api-inference.huggingface.co/models/{mname}"
    headers = {"Authorization": f"Bearer {token}"}
    class_3_models, not_class_3_models, error_models = {}, {}, {}
    try:
        response = requests.post(API_URL, headers=headers, json={
            "inputs": "I like you. I love you",
            "options": {"wait_for_model": True}
        })
        print(f"Model: {mname}, Status Code: {response.status_code}")
        output = response.json()
        print(f"Output: {output}")
        
        
        if len(output[0]) == 3:
            class_3_models = {mname: output}
        else:
            not_class_3_models = {mname: output}
    except Exception as e:
        print(f"Error for model {mname}: {e}")
        error_models = {mname: str(e)}
        
    return class_3_models, not_class_3_models, error_models

    
    

    
if __name__=='__main__' :
    pool=Pool(2)
    token_list=[
        '',
        ''
    ]
    results_nli = pool.starmap(howManyLabelModels, zip(cycle(token_list), nli_keyword_models))
    # results_non_nli = pool.starmap(howManyLabelModels, zip(cycle(token_list), nli_non_keyword_models))

    merged_class_3_models_nli = {}
    merged_not_class_3_models_nli = {}
    merged_error_models_nli = {}
    # merged_class_3_models_non_nli = {}
    # merged_not_class_3_models_non_nli = {}
    # merged_error_models_non_nli = {}

     # Merge results for models with "nli" keyword
    for class_3, not_class_3, error in tqdm(results_nli):
        merged_class_3_models_nli.update(class_3)
        merged_not_class_3_models_nli.update(not_class_3)
        merged_error_models_nli.update(error)
    
    # Merge results for models without "nli" keyword
    # for class_3, not_class_3, error in tqdm(results_non_nli):
    #     merged_class_3_models_non_nli.update(class_3)
    #     merged_not_class_3_models_non_nli.update(not_class_3)
    #     merged_error_models_non_nli.update(error)

    with open('nli_models_with_keyword_class_3.json', 'w') as file:
        json.dump(merged_class_3_models_nli, file)
    with open('nli_models_with_keyword_not_class_3.json', 'w') as file:
        json.dump(merged_not_class_3_models_nli, file)
    with open('nli_models_with_keyword_error.json', 'w') as file:
        json.dump(merged_error_models_nli, file)
    
    # with open('nli_models_without_keyword_class_3.json', 'w') as file:
    #     json.dump(merged_class_3_models_non_nli, file)
    # with open('nli_models_without_keyword_not_class_3.json', 'w') as file:
    #     json.dump(merged_not_class_3_models_non_nli, file)
    # with open('nli_models_without_keyword_error.json', 'w') as file:
    #     json.dump(merged_error_models_non_nli, file)

    