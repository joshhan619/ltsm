from ltsm.data_pipeline import TokenizerTrainingPipeline, tokenizer_get_args, tokenizer_seed_all

if __name__ == "__main__":
    args = tokenizer_get_args()
    tokenizer_seed_all(args.seed)
    pipeline = TokenizerTrainingPipeline(args)
    pipeline.run()