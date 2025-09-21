# Scenario 1:
# batch_latents (use training batch latents directly) # training_batch.latents already contains [clean prev + noisy curr] batch_input_latents = training_batch.latents prev_input_latents, prev_outout_latents = batch_input_latents.chunk(2,dim = 1) logger.info(f"Batch input latents (direct from training): {prev_input_latents.shape}") 
# logger.info(f" prev_seq_len: {training_batch.prev_seq_len}, 
# total_seq_len: {batch_input_latents.shape[1]}")


# contains [clean prev + noisy curr] batch_input_latents = training_batch.latents , 
# prev_input_latents, prev_outout_latents = batch_input_latents.chunk(2,dim = 1) logger.info(f"Batch input latents (direct from training):