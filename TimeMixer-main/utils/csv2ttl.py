import csv
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

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

# 创建一个新的RDF图
g = Graph()
g.bind("brick", BRICK)
g.bind("bf", BRICKFRAME)
g.bind("btag", BRICKTAG)
g.bind("ex", EX)

# 读取CSV文件
csv_file_path = '/home/yun/Public/Python/HVAC-Fault-Prediction/data/Data_Article_Dataset.csv'

print("Reading CSV file...")
batch_size = 10000  # 每一批次处理的行数
batch_count = 0  # 批次计数

with open(csv_file_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        entity_id = row['Datetime'].replace(" ", "_")
        entity_uri = EX[entity_id]
        
        g.add((entity_uri, RDF.type, BRICK['Measurement']))
        g.add((entity_uri, RDFS.label, Literal(entity_id, datatype=XSD.string)))
        
        for key, value in row.items():
            if key != 'Datetime':
                g.add((entity_uri, BRICK[key], Literal(value, datatype=XSD.string)))
        
        if (i + 1) % batch_size == 0:
            batch_count += 1
            batch_file_path = f'/home/yun/Public/Python/TimeMixer/ttl/brick_model_1.3_batch_{batch_count}.ttl'
            print(f"Serializing batch {batch_count} to {batch_file_path}...")
            g.serialize(destination=batch_file_path, format='turtle')
            g = Graph()  # 重置图
            g.bind("brick", BRICK)
            g.bind("bf", BRICKFRAME)
            g.bind("btag", BRICKTAG)
            g.bind("ex", EX)
            print(f"Batch {batch_count} serialized.")

print("CSV file read completed.")

# 序列化最后一批
if len(g) > 0:
    batch_count += 1
    batch_file_path = f'../brick_model_1.3_batch_{batch_count}.ttl'
    print(f"Serializing final batch {batch_count} to {batch_file_path}...")
    g.serialize(destination=batch_file_path, format='turtle')
    print(f"Final batch {batch_count} serialized.")

print("All batches serialized and process completed.")
