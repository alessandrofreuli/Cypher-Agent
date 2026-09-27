import argparse
import asyncio

from cypher_agent.runner import question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question to the Neo4j plan-and-execute agent.")
    parser.add_argument("question", nargs="+", help="The question to answer.")
    parser.add_argument("--recursion-limit", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(question(" ".join(args.question), recursion_limit=args.recursion_limit))


if __name__ == "__main__":
    main()
