import collections

def rerank(l, topk=1):
    dict_l = collections.Counter(l)
    sort_orders = sorted(dict_l.items(), key=lambda x: x[1], reverse=True)
    print('sort_orders', sort_orders)
    sort_orders = [a for a,b in sort_orders]
    res = sort_orders[:topk]
    return res
