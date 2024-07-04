import csv
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义Brick命名空间和路径
BRICK = Namespace("https://brickschema.org/schema/1.3/Brick#")
BRICKFRAME = Namespace("https://brickschema.org/schema/1.3/BrickFrame#")
BRICKTAG = Namespace("https://brickschema.org/schema/1.3/BrickTag#")
EX = Namespace("http://example.com/")

# 加载Brick 1.3的TTL文件
brick_file_path = '/home/yun/Public/Python/HVAC-Fault-Prediction/Brick建模库1.3版本Brick.ttl'
brick_graph = Graph()
print("Loading Brick 1.3 TTL file...")
brick_graph.parse(brick_file_path, format='turtle')
print("Brick 1.3 TTL file loaded.")

# 创建命名空间绑定函数
def bind_namespaces(graph):
    graph.bind("brick", BRICK)
    graph.bind("bf", BRICKFRAME)
    graph.bind("btag", BRICKTAG)
    graph.bind("ex", EX)
    return graph

# 输出目录
output_dir = '/home/yun/Public/Python/TimeMixer/ttl/'
os.makedirs(output_dir, exist_ok=True)

# 多线程处理函数
def process_batch(batch_rows, batch_num):
    g = bind_namespaces(Graph())
    for row in batch_rows:
        entity_id = row['Datetime'].replace(" ", "_")
        entity_uri = EX[entity_id]
        g.add((entity_uri, RDF.type, BRICK['Measurement']))
        g.add((entity_uri, RDFS.label, Literal(entity_id, datatype=XSD.string)))
        for key, value in row.items():
            if key != 'Datetime':
                g.add((entity_uri, BRICK[key], Literal(value, datatype=XSD.string)))
    batch_file_path = os.path.join(output_dir, f'brick_model_1.3_batch_{batch_num}.ttl')
    abs_path = os.path.abspath(batch_file_path)
    g.serialize(destination=abs_path, format='turtle')
    print(f"Batch {batch_num} serialized and saved to {abs_path}")
    return batch_num

# 读取CSV文件并分批处理
csv_file_path = '/home/yun/Public/Python/HVAC-Fault-Prediction/data/Data_Article_Dataset.csv'
batch_size = 10000  # 每一批次处理的行数

print("Reading CSV file...")
batches = []
with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    batch_rows = []
    batch_num = 0
    for i, row in enumerate(reader):
        batch_rows.append(row)
        if (i + 1) % batch_size == 0:
            batch_num += 1
            batches.append((batch_rows, batch_num))
            batch_rows = []
    if batch_rows:
        batch_num += 1
        batches.append((batch_rows, batch_num))

print(f"Total batches to process: {batch_num}")

# 使用ThreadPoolExecutor进行多线程处理
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_batch, batch[0], batch[1]) for batch in batches]
    for future in as_completed(futures):
        try:
            result = future.result()
            print(f"Batch {result} completed.")
        except Exception as e:
            print(f"Batch processing raised an exception: {e}")

print("All batches processed and serialized.")
