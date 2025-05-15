# BLADE: Brainstorming LLMs as Algorithm Designer through Evolution for Online Subset Selection

Requirements, for python=3.12: 
```python
    absl-py==2.0.0
    click==8.1
    cloudpickle==3.0.0
    openai==1.65.4
    python-dotenv==1.0.0
    scipy==1.11.4
    numpy==1.26.2
```

Install:
```cmd
pip install .
```

Settings before running:
+ metadata for LLM: build it based on the example in model_metadata.
+ make question/evaluation .py file for your own tasks: example in `./examples/` .

Usage:
+ working directory: parent of `blade`
+ cmd calling `blade run` is equal to `python blade/__main__.py run`
+ call this (you can also change the datasets and LLMs):
```cmd
python blade/__main__.py run examples/online_submodular_max_card.py \
"./examples/online_submodular_max_card/forest_cover_subset_10features_val_70.npy,./examples/online_submodular_max_card/creditcard_fraud_29features_val_70.npy,./examples/online_submodular_max_card/youtube_4features_laplacian_val_70.npy,./examples/online_submodular_max_card/twitter_dict_list_val_70.npy,./examples/online_submodular_max_card/forest_cover_subset_10features_test_30.npy,./examples/online_submodular_max_card/creditcard_fraud_29features_test_30.npy,./examples/online_submodular_max_card/youtube_4features_laplacian_test_30.npy,./examples/online_submodular_max_card/twitter_dict_list_test_30.npy" \
--validation_num 4 --model_jsons ./model_metadata/sample_llm_metadata.json,./model_metadata/sample_llm_metadata.json \
--sandbox_type ExternalProcessSandbox --brainstorming_times_per_prompt 3 --generators 20 --max_cpu_count 104 --signature_precision 4 \
--exp_name online_submodular_max_card_ds_abl
```
+ restart from breakpoint: run with `--load_backup ./data/.../backups/xxx.pickle`

Main Features:
+ Brainstorming framework.
+ KL divergence for sampling codes for prompt.
+ multi process implementation for Generator, Evaluator, CodesDatabase.
+ logging integrated. Evaluated results in `./data/{timestamp}-{exp_name}/results.log`.
+ score and signature aggrgation.
+ add brute-force extraction to get codes from response as fallback for regex failure. It requires qualified LLMs.


Developed with the help from https://github.com/google-deepmind/funsearch via https://github.com/kitft/funsearch/

---

This repository accompanies the publication

> Romera-Paredes, B. et al. [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6). *Nature* (2023)

If you use the code or data in this package, please cite:

```bibtex
@Article{FunSearch2023,
  author  = {Romera-Paredes, Bernardino and Barekatain, Mohammadamin and Novikov, Alexander and Balog, Matej and Kumar, M. Pawan and Dupont, Emilien and Ruiz, Francisco J. R. and Ellenberg, Jordan and Wang, Pengming and Fawzi, Omar and Kohli, Pushmeet and Fawzi, Alhussein},
  journal = {Nature},
  title   = {Mathematical discoveries from program search with large language models},
  year    = {2023},
  doi     = {10.1038/s41586-023-06924-6}
}
```

---

```
Copyright 2025 KurohaneReko

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.
