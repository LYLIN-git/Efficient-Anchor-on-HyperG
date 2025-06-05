from reading import read_hypergraph

for name in ["zoo", "mushroom", "covertype"]:
    path = f"data/Paper_datasets/{name}.hmetis"
    n, m, *_ = read_hypergraph(path)
    print(f"{name}: 節點數 |V| = {n}，超邊數 |E| = {m}")


n, m, * \
    _node_w, hypergraph, weights, _center_id, _hnode_w = read_hypergraph(path)
print(f"第一個超邊連接: {hypergraph[0]}")
