def test_agent_graph_compiles_without_an_llm():
    # Building the graph must not need a running Ollama server
    from agent_graph import build_graph
    graph = build_graph()
    assert graph is not None
