from ltsm.data_pipeline import TrainingPipeline, get_args, seed_all

if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)
    pipeline = TrainingPipeline(args)
    pipeline.run()