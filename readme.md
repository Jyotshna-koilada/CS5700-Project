Github link to the code: [link](https://github.com/Jyotshna-koilada/CS5700-Project.git)

The code is not entirely our own. We took the DetectGPT paper source code as base for our project. We were able to go through and understand the code, debug it and change it according to our model and convenience for our experimentation. We were able to successfully run our modified code and obtain results accordingly. Our code will take around 25 minutes for execution with default parameters.

As given in the default parameters we are using "gpt2-medium" as our base model and "t5-large" as our mask filling model. Since we didn't have the access to any server, we weren't able to reflect results with other datasets larger than gpt2-medium. gpt2-small is unfortunately not available to access.

We can change both base model and mask filling model through command line. Other prominent parameters which can be changed are dataset, percentage of masked words, no.of samples, perturbation list, batch size of data.
 
The code can be cloned using the command: !git clone https://github.com/Jyotshna-koilada/CS5700-Project.git
Once the files are cloned into the environment, we can run the code. Please check if all the necessary libraries are available to run the code like datasets, transformers, torch, numpy, matplotlib, tqdm, sklearn, os, re

The basic command line to run the program is: !python CS5700-Project/main.py

To change any arguments we need to add them as: !python CS5700-Project/main.py --{argument_name} {argument_value}
Eg., To change of no.of perturbations from 1,10 to 1,100 the command line is: !python CS5700-Project/main.py --n_pertubation_list 1,100
Please note that there shouldn't be any spaces in the values given around comma for this argument.

Similarly, other argument can be changed using their names as below:
Dataset: --dataset
Dataset key: --dataset_key
% of words masked: --pct_words_masked
Span length: --span_length
No.of samples: --n_samples
No.of perturbations list: --n_perturbation_list
Base model: --base_model_name
Mask filling model: --mask_filling_model_name
Batch size: --batch_size
Chunk size: --chunk_size
Cache directory: --cache_dir

The results (ROC curves, LL & LLR Histograms) are saved in the folder "results/" 
