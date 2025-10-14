torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name ND --paraphrases "same_order" --num_train_epochs 128 --eval_epoch 0 1 2 4 8 16 32 64 96 128  --save_checkpoint 1 --save_path ./results/ar/

torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name ND --paraphrases "None" --num_train_epochs 3840 --eval_epoch 0 30 60 120 240 480 960 1920 2880 3840 --save_checkpoint 1 --save_path ./results/ar/


torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name biography --paraphrases "same_order" --num_train_epochs 1024 --eval_epoch 0 1 2 4 8 16 32 64 128 256 512 768 1024  --save_checkpoint 1 --save_path ./results/ar/

torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name biography --paraphrases "change_order" --num_train_epochs 1024 --eval_epoch 0 1 2 4 8 16 32 64 128 256 512 768 1024  --save_checkpoint 1 --save_path ./results/ar/

torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name biography --paraphrases "None" --num_train_epochs 5120 --eval_epoch 0 5 10 20 40 80 160 320 640 1280 2560 3840 5120  --save_checkpoint 1 --save_path ./results/ar/


torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name wiki --paraphrases "same_order" --num_train_epochs 512 --eval_epoch 0 1 2 4 8 16 32 64 128 256 384 512  --save_checkpoint 1 --save_path ./results/ar/

torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name wiki --paraphrases "change_order" --num_train_epochs 512 --eval_epoch 0 1 2 4 8 16 32 64 128 256 384 512  --save_checkpoint 1 --save_path ./results/ar/

torchrun --nproc_per_node=4 ../src/ar.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --dataset_name wiki --paraphrases "None" --num_train_epochs 5120 --eval_epoch 0 10 20 40 80 160 320 640 1280 2560 3840 5120  --save_checkpoint 1 --save_path ./results/ar/