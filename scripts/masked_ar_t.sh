torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0.5 --dataset_name ND --paraphrases "None" --num_train_epochs 2880 --eval_epoch 0 30 60 120 240 480 960 1920 2880 --save_checkpoint 1 --save_path ./results/masked_ar_t/

torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0.5 --dataset_name ND --paraphrases "same_order" --num_train_epochs 96 --eval_epoch 0 1 2 4 8 16 32 64 96  --save_checkpoint 1 --save_path ./results/masked_ar_t/


torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 1 --dataset_name ND --paraphrases "None" --num_train_epochs 2880 --eval_epoch 0 30 60 120 240 480 960 1920 2880 --save_checkpoint 1 --save_path ./results/masked_ar_t/

torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 1 --dataset_name ND --paraphrases "same_order" --num_train_epochs 96 --eval_epoch 0 1 2 4 8 16 32 64 96  --save_checkpoint 1 --save_path ./results/masked_ar_t/


torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0 --dataset_name ND --paraphrases "None" --num_train_epochs 2880 --eval_epoch 0 30 60 120 240 480 960 1920 2880 --save_checkpoint 1 --save_path ./results/masked_ar_t/

torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0 --dataset_name ND --paraphrases "same_order" --num_train_epochs 96 --eval_epoch 0 1 2 4 8 16 32 64 96  --save_checkpoint 1 --save_path ./results/masked_ar_t/


torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0.75 --dataset_name ND --paraphrases "None" --num_train_epochs 2880 --eval_epoch 0 30 60 120 240 480 960 1920 2880 --save_checkpoint 1 --save_path ./results/masked_ar_t/

torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0.75 --dataset_name ND --paraphrases "same_order" --num_train_epochs 96 --eval_epoch 0 1 2 4 8 16 32 64 96  --save_checkpoint 1 --save_path ./results/masked_ar_t/


torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0.25 --dataset_name ND --paraphrases "None" --num_train_epochs 2880 --eval_epoch 0 30 60 120 240 480 960 1920 2880 --save_checkpoint 1 --save_path ./results/masked_ar_t/

torchrun --nproc_per_node=4 ../src/masked_ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --t_mode "fixed" --t 0.25 --dataset_name ND --paraphrases "same_order" --num_train_epochs 96 --eval_epoch 0 1 2 4 8 16 32 64 96  --save_checkpoint 1 --save_path ./results/masked_ar_t/
