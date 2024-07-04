for bs in 1 2 4 8 16 32 64 128 256 512
do
    python3 prob_of_selection.py --n_samples $bs --solver random --max_batch_size $bs --n_experts 160 --k_experts_each 6 --k_candidate_experts 6
done