"""Live pipeline verification script."""

import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import VideoRetrievalPipeline

def main():
    print("=" * 70)
    print("Starting Live Educational Video Retrieval Pipeline Verification...")
    print("=" * 70)

    pipeline = VideoRetrievalPipeline()
    query = "What is a neural network and how do neurons work?"
    print(f"Query: '{query}'")

    start = time.time()
    results_data = pipeline.process_query(query, top_k=6)
    elapsed = time.time() - start

    print("\nPipeline Summary:")
    print(f"- Total Candidate Videos: {results_data['total_candidates']}")
    print(f"- Total Timestamped Chunks: {results_data['total_chunks']}")
    print(f"- Total Ranked Results: {results_data['total_results']}")
    print(f"- Execution Time: {results_data['execution_time_seconds']:.2f}s (Total: {elapsed:.2f}s)")
    print("\n" + "=" * 70)
    print("TOP RANKED EDUCATIONAL LECTURE SEGMENTS:")
    print("=" * 70)

    for idx, item in enumerate(results_data["results"], start=1):
        print(f"\n#{idx}: {item['title']}")
        print(f"    Channel:       {item['channel_name']}")
        print(f"    Relevance:     {item['relevance_score']:.4f}")
        print(f"    Timestamp:     {item['start_time_formatted']} - {item['end_time_formatted']} ({item['start_time_seconds']}s - {item['end_time_seconds']}s)")
        print(f"    Timestamp URL: {item['timestamp_url']}")
        print(f"    Excerpt:       \"{item['topic_text'][:140]}...\"")

    print("\n" + "=" * 70)
    print("Verification Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
