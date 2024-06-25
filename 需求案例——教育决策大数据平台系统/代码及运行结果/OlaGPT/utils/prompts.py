# -*- coding: utf-8 -*-

def get_query_format(dataset, query_json):
    # intention refinement and enhancement and
    if dataset == 'aqua':
        query_json['instruct'] = f"现在给你一个{query_json['llm_task_type']} 的问题与它们选择，答案要以json的形式返回:"
    elif dataset == 'ekar_chinese':
        query_json['instruct'] = "现在给你一个类比问题和选择，答案要以json的形式返回:"

    prefix = """请注意！选出的答案必须以json格式结束: {Answer: one of options[A,B,C,D,E]}."""

    final_query = '\n'.join([
        query_json['instruct'],
        query_json['context'],
        query_json['question'],
        query_json['options'],
        prefix
    ])

    return query_json, final_query
