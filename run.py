import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [api|train|evaluate|benchmark]")
        sys.exit(1)
    command = sys.argv[1]
    args = sys.argv[2:]
    if command == "api":
        import uvicorn
        from backend.main import create_app
        from core.config import get_settings

        settings = get_settings()
        app = create_app()
        uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
    elif command == "train":
        from scripts.train import main as train_main

        sys.argv = ["train"] + args
        train_main()
    elif command == "evaluate":
        from scripts.evaluate import main as eval_main

        sys.argv = ["evaluate"] + args
        eval_main()
    elif command == "benchmark":
        from scripts.benchmark import main as bench_main

        sys.argv = ["benchmark"] + args
        bench_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
