n_clients=8
# 使用 $(seq 1 $n_clients) 来生成从1到n_clients的序列
for id in $(seq 1 $n_clients)
do
  client_local_dir=workspaces/secure_workspace/site-${id}/local
  cp ${client_local_dir}/resources.json.default ${client_local_dir}/resources.json
  # 确保 N_GPU 这个变量已经被定义和设置过
  sed -i "s|\"num_of_gpus\": 0|\"num_of_gpus\": ${N_GPU}|g" ${client_local_dir}/resources.json
  sed -i "s|\"mem_per_gpu_in_GiB\": 0|\"mem_per_gpu_in_GiB\": 1|g" ${client_local_dir}/resources.json
done