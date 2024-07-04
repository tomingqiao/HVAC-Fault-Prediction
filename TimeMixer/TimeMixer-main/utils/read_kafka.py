# 导入必要的库
from confluent_kafka import Consumer, KafkaError  # 用于与 Kafka 进行交互
from rdflib import Graph  # 用于处理 RDF 数据

# Kafka 消费者配置
conf = {
    'bootstrap.servers': 'localhost:9092',  # Kafka broker 地址
    'group.id': 'mygroup',  # 消费者组 ID
    'auto.offset.reset': 'earliest'  # 从最早的偏移量开始消费
}

# 创建 Kafka 消费者
consumer = Consumer(conf)

# 指定要消费的主题
topic = 'my_topic'
consumer.subscribe([topic])  # 订阅主题

# 定义保存 TTL 文件的函数
def save_ttl_file(data, filename='received_data.ttl'):
    """
    将数据保存到 TTL 文件中。

    Args:
        data (str): 要保存的数据。
        filename (str, optional): 文件名。默认为 'received_data.ttl'。
    """
    with open(filename, 'w', encoding='utf-8') as f:  # 打开文件以进行写入
        f.write(data)  # 将数据写入文件


# 主循环，用于消费消息
try:
    while True:
        # 从 Kafka 主题中获取消息
        msg = consumer.poll(timeout=1.0)  # 设置超时时间为 1 秒

        # 如果没有消息，则继续循环
        if msg is None:
            continue

        # 如果消息有错误，则打印错误信息
        if msg.error():
            # 如果是分区末尾的错误，则继续循环
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            # 否则，打印错误信息并退出循环
            else:
                print(msg.error())
                break

        # 获取消息并保存为 TTL 文件
        data = msg.value().decode('utf-8')  # 将字节类型的消息转换为字符串
        save_ttl_file(data)  # 将数据保存到 TTL 文件


# 处理键盘中断异常
except KeyboardInterrupt:
    pass

# 最后关闭消费者
finally:
    consumer.close()