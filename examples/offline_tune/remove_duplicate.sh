#!/bin/bash
file_name_list=(dump_gemm_shapes.txt)
for file in ${file_name_list[@]}; do
    echo "Processing $file"
    sed -E 's/--solution_index [0-9]+//g; s/--algo_method index//g; s/--cold_iters 0/--cold_iters 100/g; s/--iters 0/--iters 100/g; s/--rotating 0/--rotating 512/g' $file | sort | uniq | awk '{print $0, "--skip_slow_solution_ratio 0.7"}' > unique_${file} 
done
