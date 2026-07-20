import os
import sys
import argparse
from agentic_graph_rag import AgenticGraphRAGEngine

def main():
    parser = argparse.ArgumentParser(description="SEC Corporate Subsidiary Agentic Graph RAG Tool")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The natural language question to ask about the SEC filing data"
    )
    args = parser.parse_args()

    graph_file = "data_graph.ttl"
    if not os.path.exists(graph_file):
        print(f"Error: {graph_file} not found. Please run the extraction script first: python scripts/extract_sec_data.py")
        sys.exit(1)

    print("Initializing Agentic Graph RAG Engine...")
    engine = AgenticGraphRAGEngine(graph_file)
    
    print("\nProcessing prompt through Agentic Pipeline...")
    result = engine.query(args.query)
    
    print(result)

if __name__ == "__main__":
    main()
