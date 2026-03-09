"""
main.py

Entry point and interactive CLI for the Supervisor agent.

Usage:
    python -m Supervisor.main                     # interactive REPL
    python -m Supervisor.main --query "سؤال"      # single query
    python -m Supervisor.main --case-id 123       # set active case
"""

import argparse
import json
import logging
import sys
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from Supervisor.config import MAX_RETRIES
from Supervisor.graph import app
from Supervisor.state import SupervisorState


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _default_state(
    query: str,
    case_id: str = "",
    uploaded_files: List[str] | None = None,
    conversation_history: List[dict] | None = None,
    turn_count: int = 0,
) -> SupervisorState:
    """Build a minimal initial state for a supervisor invocation."""
    return SupervisorState(
        judge_query=query,
        case_id=case_id,
        uploaded_files=uploaded_files or [],
        conversation_history=conversation_history or [],
        turn_count=turn_count,
        intent="",
        target_agents=[],
        classified_query="",
        agent_results={},
        agent_errors={},
        validation_status="",
        validation_feedback="",
        retry_count=0,
        max_retries=MAX_RETRIES,
        document_classifications=[],
        merged_response="",
        final_response="",
        sources=[],
    )


def run_single_query(
    query: str,
    case_id: str = "",
    uploaded_files: Optional[List[str]] = None,
    conversation_history: Optional[List[dict]] = None,
    turn_count: int = 0,
) -> dict:
    """Run a single query through the supervisor and return the final state."""
    state = _default_state(
        query=query,
        case_id=case_id,
        uploaded_files=uploaded_files,
        conversation_history=conversation_history,
        turn_count=turn_count,
    )
    result = app.invoke(state)
    return result


def interactive_loop(case_id: str = "") -> None:
    """Start an interactive REPL for conversational judge queries."""
    print("=" * 60)
    print("  Supervisor Agent - Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    conversation_history: List[dict] = []
    turn_count = 0

    while True:
        try:
            query = input("\n[Judge]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        result = run_single_query(
            query=query,
            case_id=case_id,
            conversation_history=conversation_history,
            turn_count=turn_count,
        )

        final = result.get("final_response", "(no response)")
        sources = result.get("sources", [])

        print(f"\n[Assistant]: {final}")
        if sources:
            print(f"\n  Sources: {', '.join(sources)}")

        # Carry forward conversation state
        conversation_history = result.get("conversation_history", [])
        turn_count = result.get("turn_count", turn_count + 1)


def ingest_files(
    file_paths: List[str],
    case_id: str = "",
) -> List[dict]:
    """Ingest files into MongoDB and the vector store *before* running a query.

    This is the recommended way to add documents to a case ahead of time.
    Supports text files, PDFs, and images (images go through OCR first).

    Parameters
    ----------
    file_paths : list of str
        Paths to files on disk.
    case_id : str
        The case these documents belong to.

    Returns
    -------
    list of dict
        One classification/storage result per file.
    """
    from Supervisor.services.file_ingestor import FileIngestor
    from Supervisor.config import (
        MONGO_URI, MONGO_DB, MONGO_COLLECTION,
        EMBEDDING_MODEL, CHROMA_COLLECTION, CHROMA_PERSIST_DIR,
    )

    ingestor = FileIngestor(
        mongo_uri=MONGO_URI,
        mongo_db=MONGO_DB,
        mongo_collection=MONGO_COLLECTION,
        embedding_model=EMBEDDING_MODEL,
        chroma_collection=CHROMA_COLLECTION,
        chroma_persist_dir=CHROMA_PERSIST_DIR,
    )
    return ingestor.ingest_files(file_paths, case_id=case_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor Agent CLI")
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single query to run (non-interactive mode)",
    )
    parser.add_argument(
        "--case-id", "-c",
        type=str,
        default="",
        help="Active case identifier",
    )
    parser.add_argument(
        "--files", "-f",
        nargs="*",
        default=None,
        help="Uploaded file paths for OCR processing",
    )
    parser.add_argument(
        "--ingest", "-i",
        nargs="*",
        default=None,
        help=(
            "Ingest files into the system before running any query. "
            "Supports text, PDF, and image files."
        ),
    )
    args = parser.parse_args()

    # Pre-ingest files if requested
    if args.ingest:
        logger.info("Ingesting %d file(s)...", len(args.ingest))
        results = ingest_files(args.ingest, case_id=args.case_id)
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        if not args.query:
            return

    if args.query:
        result = run_single_query(
            query=args.query,
            case_id=args.case_id,
            uploaded_files=args.files,
        )
        print(json.dumps(
            {
                "intent": result.get("intent"),
                "target_agents": result.get("target_agents"),
                "final_response": result.get("final_response"),
                "sources": result.get("sources"),
                "validation_status": result.get("validation_status"),
            },
            ensure_ascii=False,
            indent=2,
        ))
    else:
        interactive_loop(case_id=args.case_id)


if __name__ == "__main__":
    main()
