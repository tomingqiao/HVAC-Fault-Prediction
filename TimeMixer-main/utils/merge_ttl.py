from rdflib import Graph

# 创建一个新的RDF图用于合并
merged_graph = Graph()

# 合并所有批次文件
batch_count = 0
while True:
    batch_file_path = f'/home/yun/Public/Python/TimeMixer/ttl/brick_model_1.3_batch_{batch_count + 1}.ttl'
    try:
        print(f"Loading {batch_file_path}...")
        merged_graph.parse(batch_file_path, format='turtle')
        batch_count += 1
    except FileNotFoundError:
        print(f"File {batch_file_path} not found, stopping.")
        break

# 将合并后的RDF图序列化为一个TTL文件
merged_ttl_file_path = '/home/yun/Public/Python/TimeMixer//brick_model_1.3_merged.ttl'
print(f"Serializing merged RDF graph to {merged_ttl_file_path}...")
merged_graph.serialize(destination=merged_ttl_file_path, format='turtle')
print(f"Merged TTL file has been saved to {merged_ttl_file_path}")
