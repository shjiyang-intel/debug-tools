```
python dynamic_to_static.py -m {model_path} -o {output_dir} --inputs "a[1,2,3],b[3,4,5]"
```


# Partition

```
python partition.py -m {model_ir_path} -o output_node_name_0 output_node_name_1 -out_dir {out_dir} -n {new_model_name}

// Or
python partition.py -m {model_ir_path} -s breakpoint_node_name_0 breakpoint_node_name_1 -out_dir {out_dir} -n {new_model_name}
```